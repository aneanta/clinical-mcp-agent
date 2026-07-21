import re
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.mcp.server import (
    PATIENT_REGISTRY,
    get_patient_demographics,
    get_patient_vitals,
    get_patient_allergies,
    get_patient_medications,
    get_patient_documents
)

# Secuencia oficial de letras del DNI/NIE (módulo 23)
DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

def validate_dni_nie(doc: str, strict: bool = False) -> tuple[bool, str]:
    """Valida un DNI o NIE español. Si strict=False, permite DNIs de prueba."""
    doc = doc.upper().strip()
    
    if re.match(r'^PAT-\d{4}$', doc) or doc in ["12345", "593892", "EXAMPLE"]:
        return True, doc

    match_dni = re.match(r'^(\d{8})([A-Z])$', doc)
    if match_dni:
        num, letter = match_dni.groups()
        expected_letter = DNI_LETTERS[int(num) % 23]
        if letter == expected_letter or not strict:
            return True, doc
        return False, f"La letra '{letter}' no corresponde al número {num}."

    match_nie = re.match(r'^([XYZ])(\d{7})([A-Z])$', doc)
    if match_nie:
        prefix, num, letter = match_nie.groups()
        prefix_val = {"X": "0", "Y": "1", "Z": "2"}[prefix]
        full_num = int(prefix_val + num)
        expected_letter = DNI_LETTERS[full_num % 23]
        if letter == expected_letter or not strict:
            return True, doc
        return False, f"La letra '{letter}' no es correcta para el NIE {doc}."

    return False, "Formato no válido."


def find_documents_in_text(text: str) -> list[str]:
    patterns = [
        r'\bPAT-\d{4}\b',             
        r'\b[XYZxyz]\d{7}[A-Za-z]\b',  
        r'\b\d{8}[A-Za-z]\b',          
        r'\b12345\b',                  
        r'\b593892\b',                 
    ]
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend([f.upper() for f in found])
    return matches


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: State):
    last_user_msg = state["messages"][-1].content.strip()
    msg_lower = last_user_msg.lower()

    # 1. Gestionar cambio de paciente
    reset_keywords = ["cambiar paciente", "cambio de paciente", "otro paciente", "nuevo paciente", "cerrar sesion", "otro dni"]
    if any(kw in msg_lower for kw in reset_keywords):
        response_text = (
            "Sesión de paciente finalizada.\n\n"
            "Por favor, introduce el DNI o NIE del nuevo paciente para comenzar a consultar su expediente:"
        )
        return {"messages": [AIMessage(content=response_text)]}

    # 2. Buscar DNI activo en el historial
    active_doc = None
    all_user_messages = [m.content for m in state["messages"] if not isinstance(m, AIMessage)]
    
    for msg in reversed(all_user_messages):
        candidates = find_documents_in_text(msg)
        for cand in candidates:
            is_valid, _ = validate_dni_nie(cand, strict=False)
            if is_valid:
                active_doc = cand
                break
        if active_doc:
            break

    # 3. Caso: No hay paciente activo
    if not active_doc:
        candidates = find_documents_in_text(last_user_msg)
        if candidates:
            _, val_detail = validate_dni_nie(candidates[0], strict=True)
            response_text = (
                f"DNI / NIE No Válido ({candidates[0]})\n\n"
                f"Motivo: {val_detail}\n\n"
                f"Por favor, verifica los dígitos e introduce de nuevo un DNI o NIE válido:"
            )
        else:
            response_text = (
                "Autenticación de Paciente Requerida\n\n"
                "Por favor, introduce un DNI/NIE válido (Ejemplo: 12345678Z, Y1234567Z o PAT-8832):"
            )
        return {"messages": [AIMessage(content=response_text)]}

    # 4. Obtener datos básicos del paciente
    patient_id = PATIENT_REGISTRY.get(active_doc, active_doc)
    try:
        demo = get_patient_demographics(patient_id)
        given_name = demo.get('name', [{}])[0].get('given', ['Paciente'])[0]
        family_name = demo.get('name', [{}])[0].get('family', 'FHIR')
        nombre_paciente = f"{given_name} {family_name}"
        birth_date = demo.get('birthDate', 'N/A')
    except Exception:
        nombre_paciente = "Paciente FHIR"
        birth_date = "N/A"

    # ==========================================
    # LÓGICA RÁPIDA (SOLO CONSULTA LO NECESARIO)
    # ==========================================

    # CASO A: Pide información clínica -> Carga constantes/alergias/medicación
    if any(k in msg_lower for k in ["constante", "vital", "presion", "alergia", "intolerancia", "medicacion", "tratamiento", "farmaco", "receta", "paciente", "ficha", "resumen", "fhir"]):
        try:
            vitals = get_patient_vitals(patient_id) or []
            allergies = get_patient_allergies(patient_id) or []
            meds = get_patient_medications(patient_id) or []
        except Exception:
            vitals, allergies, meds = [], [], []

        str_allergies = "".join([f"- {a.get('code', {}).get('text', 'N/A')} (Criticidad: {str(a.get('criticality', 'N/A')).upper()})\n" for a in allergies])
        str_meds = "".join([f"- {(m.get('medication', {}).get('concept', {}) or m.get('medicationCodeableConcept', {})).get('text', 'N/A')} ({m.get('dosageInstruction', [{}])[0].get('text', 'Sin pauta')})\n" for m in meds])
        str_vitals = "".join([f"- {v.get('code', {}).get('text', 'N/A')}: {v.get('valueQuantity', {}).get('value', 'N/A')} {v.get('valueQuantity', {}).get('unit', '')}\n" for v in vitals])

        text_allergies = str_allergies if str_allergies else "- Ninguna registrada\n"
        text_meds = str_meds if str_meds else "- Sin tratamientos activos\n"
        text_vitals = str_vitals if str_vitals else "- Sin constantes registradas"

        response_text = (
            "Expediente Clínico FHIR\n"
            f"Paciente: {nombre_paciente} | DNI: {active_doc} | Nacimiento: {birth_date}\n\n"
            f"Alergias Registradas:\n{text_allergies}\n"
            f"Medicación Activa:\n{text_meds}\n"
            f"Últimas Constantes Vitales:\n{text_vitals}"
        )
        return {"messages": [AIMessage(content=response_text)]}

    # CASO B: Pide pruebas/imágenes -> Carga SOLO los documentos en este momento
    elif any(k in msg_lower for k in ["prueba", "resonancia", "imagen", "foto", "pdf", "informe", "analitica"]):
        try:
            docs = get_patient_documents(patient_id) or []
        except Exception:
            docs = []

        if docs:
            response_text = f"Pruebas Médicas — {nombre_paciente} (DNI: {active_doc}):\n\nAquí tienes los archivos disponibles. He cargado la prueba más reciente en el visor lateral."
            msg = AIMessage(content=response_text, additional_kwargs={"documents": docs})
            return {"messages": [msg]}
        else:
            response_text = f"El paciente {nombre_paciente} no tiene resonancias, archivos PDF ni fotografías subidas a su historial en este momento."
            return {"messages": [AIMessage(content=response_text)]}

    # CASO C: Acreditación o mensajes no reconocidos -> Devuelve el MENÚ inmediatamente
    else:
        response_text = (
            f"Paciente activo: {nombre_paciente} (DNI: {active_doc}).\n\n"
            f"Puedes pedirme:\n"
            f"- 'Constantes vitales'\n"
            f"- 'Alergias'\n"
            f"- 'Medicación'\n"
            f"- 'Ficha completa'\n"
            f"- 'Muéstrame sus pruebas/resonancias'\n"
            f"- 'Cambiar paciente'"
        )
        return {"messages": [AIMessage(content=response_text)]}

workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

memory = MemorySaver()
clinical_agent = workflow.compile(checkpointer=memory)