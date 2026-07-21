import os
import json
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from app.mcp.server import (
    PATIENT_REGISTRY,
    get_patient_demographics,
    get_patient_vitals,
    get_patient_allergies,
    get_patient_medications,
    get_patient_documents,
)

# ==========================================
# 1. FACTORÍA DE LLM (ABSTRACCIÓN MULTI-MOTOR)
# ==========================================
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(f"Proveedor LLM no reconocido: {provider}")


# ==========================================
# 2. HERRAMIENTAS MCP ENVOLTORIO
# ==========================================
def _format_tool_response(data):
    """Auxiliar para asegurar que el contenido devuelto sea un string JSON valido."""
    if data is None:
        return "[]"
    if isinstance(data, (dict, list)):
        # default=str convierte objetos tipo 'date' y 'datetime' a texto automáticamente
        return json.dumps(data, ensure_ascii=False, default=str)
    return str(data)

@tool
def tool_get_demographics(patient_id: str) -> str:
    """Obtiene los datos demográficos básicos del paciente (nombre, nacimiento) usando su ID o DNI."""
    data = get_patient_demographics(patient_id)
    return _format_tool_response(data)

@tool
def tool_get_vitals(patient_id: str) -> str:
    """Obtiene las constantes vitales del paciente (presión arterial, pulso, etc.)."""
    data = get_patient_vitals(patient_id)
    return _format_tool_response(data)

@tool
def tool_get_allergies(patient_id: str) -> str:
    """Obtiene las alergias e intolerancias registradas del paciente."""
    data = get_patient_allergies(patient_id)
    return _format_tool_response(data)

@tool
def tool_get_medications(patient_id: str) -> str:
    """Obtiene la medicación activa y tratamientos prescritos al paciente."""
    data = get_patient_medications(patient_id)
    return _format_tool_response(data)

@tool
def tool_get_documents(patient_id: str) -> str:
    """Obtiene las pruebas médicas, resonancias, analíticas y documentos en PDF del paciente."""
    data = get_patient_documents(patient_id)
    return _format_tool_response(data)

tools = [
    tool_get_demographics,
    tool_get_vitals,
    tool_get_allergies,
    tool_get_medications,
    tool_get_documents
]

# ==========================================
# 3. ESTADO Y PROMPT DEL SISTEMA
# ==========================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(content="""
Eres 'TFM Clinical Assistant', un asistente médico experto conectado a servidores FHIR mediante el protocolo MCP.

REGLAS DE FLUJO DE TRABAJO (ESTRICTAS):
1. NO ASUMIR PACIENTE: Si el usuario saluda (ej. "hola"), hace preguntas generales o NO indica explícitamente un DNI o ID de paciente, PROHIBIDO ejecutar herramientas.
2. PEDIR IDENTIFICADOR: Cuando falte el identificador del paciente, responde amablemente solicitando al usuario que te facilite el DNI/NIE o ID FHIR del paciente sobre el que desea consultar.
3. EJECUCIÓN DE HERRAMIENTAS: Solamente ejecuta las herramientas FHIR cuando tengas un identificador de paciente claro proporcionado en la conversación.
                              
REGLAS DE FORMATO:
1. PROHIBIDO USAR ASTERISCOS: No utilices jamás los caracteres '*' ni '**' en ninguna parte de la respuesta.
2. Formatea los títulos de las secciones en MAYÚSCULAS con un espacio superior para separarlos bien.
3. Utiliza la viñeta '• ' para listar elementos o parámetros.
4. Muestra la información directa, profesional y sin textos de relleno o avisos genéricos al final.
                              
EJEMPLO DE ESTRUCTURA DESEADA:

INFORMACIÓN DEL PACIENTE
Nombre: María García López
DNI: 12345678Z
Fecha de nacimiento: 12/04/1985

CONSTANTES VITALES
• Tensión arterial: 128/82 mmHg
• Frecuencia cardíaca: 72 bpm
• Glucosa en sangre: 95 mg/dL

ALERGIAS REGISTRADAS
• Penicilina (Gravedad alta)
• Polen de olivo (Gravedad baja)

MEDICACIÓN ACTIVA
• No hay registros de medicación activa.
""")

def call_model(state: State):
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def post_process_documents(state: State):
    """Extrae automáticamente los documentos para enviarlos al visor lateral de Next.js."""
    last_msg = state["messages"][-1]
    docs_found = []
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "tool_get_documents":
            try:
                docs_found = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except Exception:
                docs_found = []
            break

    if docs_found and isinstance(last_msg, AIMessage):
        last_msg.additional_kwargs["documents"] = docs_found

    return state

# ==========================================
# 4. CONSTRUCCIÓN DEL GRAFO LANGGRAPH
# ==========================================
workflow = StateGraph(State)

# Definimos los nodos
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("post_process", post_process_documents)

# Punto de entrada
workflow.add_edge(START, "agent")

# Mapeo condicional explícito:
# - Si la IA decide usar una herramienta -> va a "tools"
# - Si la IA decide terminar (END) -> se redirige a "post_process"
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: "post_process"
    }
)

# Bucle de retorno de las herramientas al agente
workflow.add_edge("tools", "agent")

# Salida final desde post_process
workflow.add_edge("post_process", END)

memory = MemorySaver()
clinical_agent = workflow.compile(checkpointer=memory)