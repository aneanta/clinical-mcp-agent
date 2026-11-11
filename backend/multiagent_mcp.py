import os
import asyncio
import sys
import json
import re
import time


from pathlib import Path
from typing import TypedDict, Literal

from groq import Groq, RateLimitError, BadRequestError

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from langgraph.graph import StateGraph, START, END

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.pretty import Pretty
from rich.markdown import Markdown
from rich.rule import Rule

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURACIÓN
# ============================================================

console = Console()

groq_client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

GROQ_MODEL = "qwen/qwen3.8-27b"
PLANNER_MODEL = "openai/gpt-oss-20b"
CLINICAL_MODEL = "openai/gpt-oss-20b"

def groq_completion_with_retry(**kwargs):
    max_attempts = 3

    # Opción interna nuestra. Se elimina para no enviársela a Groq.
    fallback_to_json_object = kwargs.pop(
        "fallback_to_json_object",
        False
    )

    if kwargs.get("model") == "qwen/qwen3.8-27b":
        current_max_tokens = kwargs.get(
            "max_completion_tokens"
        )

        if (
            current_max_tokens is None
            or current_max_tokens > 900
        ):
            kwargs["max_completion_tokens"] = 900

    for attempt in range(max_attempts):
        try:
            return groq_client.chat.completions.create(
                **kwargs
            )

        except BadRequestError as e:

            # Si Groq falla intentando validar el JSON Schema,
            # hacemos fallback automático a JSON Object Mode.
            if (
                fallback_to_json_object
                and "json_validate_failed" in str(e)
                and kwargs.get("response_format", {}).get("type")
                == "json_schema"
            ):
                console.print(
                    "[yellow]"
                    "Groq no ha podido generar el JSON Schema estricto. "
                    "Reintentando automáticamente con JSON Object Mode..."
                    "[/yellow]"
                )

                kwargs["response_format"] = {
                    "type": "json_object"
                }

                # Solo hacemos este fallback una vez.
                fallback_to_json_object = False

                continue

            raise

        except RateLimitError:
            if attempt == max_attempts - 1:
                raise

            wait_seconds = 3 * (attempt + 1)

            console.print(
                f"[yellow]Límite temporal de Groq alcanzado. "
                f"Reintentando la misma petición...[/yellow]"
            )

            time.sleep(wait_seconds)

# ============================================================
# STATE
# ============================================================

class RouterState(TypedDict):
    question: str
    route: str
    required_data: list[str]
    patient_context: dict
    missing_data: list[str]
    validation_ok: bool
    retry_count: int
    symptom_analysis: dict
    symptom_validation_ok: bool
    symptom_retry_count: int
    clinical_analysis: dict
    clinical_validation_ok: bool
    clinical_validation_issues: list[str]
    clinical_retry_count: int
    clinical_structure_retry_count: int
    semantic_validation_ok: bool
    semantic_validation_issues: list[str]
    response: str

# ============================================================
# ORQUESTADOR
# ============================================================

def router(state: RouterState):

    question = state["question"]

    response = groq_completion_with_retry(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres el orquestador de un sistema multiagente médico. "
                    "Tu única tarea es decidir qué ruta debe seguir la petición. "

                    "Debes responder EXACTAMENTE con una de estas tres opciones: "
                    "'datos', 'clinico' o 'datos_y_clinico'. "

                    "REGLAS: "

                    "1. Usa 'datos' cuando el usuario solo quiera consultar "
                    "información existente de un paciente, por ejemplo su edad, "
                    "medicación o resultados de laboratorio. "

                    "2. Usa 'clinico' cuando el usuario plantee síntomas o una "
                    "cuestión clínica general que pueda responderse sin consultar "
                    "información almacenada de un paciente concreto. "

                    "3. Usa 'datos_y_clinico' cuando la petición mencione un paciente "
                    "concreto y sea necesario utilizar sus datos almacenados para "
                    "realizar posteriormente razonamiento clínico. "
                    "También debes usar 'datos_y_clinico' si el usuario pide "
                    "explícitamente tener en cuenta, consultar, revisar o utilizar "
                    "los datos de un paciente para responder. "

                    "EJEMPLOS: "

                    "'¿Qué medicación toma el paciente 482?' -> datos. "

                    "'Un paciente presenta sed y fatiga. ¿Qué información sería "
                    "útil recopilar?' -> clinico. "

                    "'El paciente 482 presenta sed y fatiga. Teniendo en cuenta "
                    "sus datos, ¿qué información sería útil recopilar?' "
                    "-> datos_y_clinico. "

                    "No expliques tu decisión. "
                    "Devuelve solamente el nombre de la ruta."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    route = response.choices[0].message.content.strip().lower()

    print("DEBUG ROUTER:", route)
    # VISUALIZACIÓN DEL ROUTER
    route_names = {
        "datos": "Agente de datos",
        "clinico": "Agente clínico",
        "datos_y_clinico": "Datos → Clínico"
    }

    console.print()
    console.print(
        Panel(
            f"[bold]Ruta seleccionada:[/bold] "
            f"[cyan]{route_names.get(route, route)}[/cyan]\n\n"
            f"[dim]Código de ruta: {route}[/dim]",
            title="🧠 ORQUESTADOR",
            border_style="cyan"
        )
    )

    return {
        "route": route
    }

# ============================================================
# Planificador
# ============================================================

def planner(state: RouterState):

    console.print()
    console.print(
        Panel(
            "[bold]Planificando qué datos son necesarios...[/bold]",
            title="🗺️ PLANNER",
            border_style="yellow"
        )
    )

    response = groq_completion_with_retry(
        model=PLANNER_MODEL,
        reasoning_effort="low",
        reasoning_format="hidden",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "planner_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "required_data": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "get_patient",
                                    "get_medications",
                                    "get_lab_results"
                                ]
                            }
                        }
                    },
                    "required": [
                        "required_data"
                    ],
                    "additionalProperties": False
                }
            }
        },

        messages=[
            {
                "role": "system",
                "content": (
                    "Eres el planificador de un sistema multiagente médico. "
                    "Tu única tarea es decidir qué fuentes de datos son estrictamente necesarias "
                    "para responder a la petición del usuario. "

                    "FUENTES DISPONIBLES: "

                    "get_patient contiene únicamente datos generales del paciente, "
                    "como nombre y edad. "

                    "get_medications contiene la medicación actual del paciente. "

                    "get_lab_results contiene los resultados de laboratorio del paciente. "

                    "REGLAS DE SELECCIÓN: "

                    "1. Selecciona únicamente las fuentes estrictamente necesarias para la consulta. "
                    "No recuperes información por si pudiera resultar útil. "

                    "2. El hecho de que la consulta mencione un paciente concreto o incluya "
                    "su identificador NO significa que debas seleccionar get_patient. "

                    "3. Selecciona get_patient solamente cuando para responder sea relevante "
                    "conocer datos generales como la edad o el nombre del paciente. "

                    "4. Selecciona get_medications cuando el usuario pregunte por la medicación, "
                    "pida tenerla en cuenta, pregunte por posibles efectos relacionados con ella "
                    "o cuando la medicación sea necesaria para responder a la consulta. "

                    "5. Selecciona get_lab_results cuando el usuario pregunte por análisis, "
                    "resultados de laboratorio, valores analíticos o pida explícitamente "
                    "tener en cuenta sus resultados de laboratorio. "

                    "6. NO selecciones get_lab_results automáticamente solo porque la consulta "
                    "contenga síntomas o requiera razonamiento clínico. "

                    "7. Si el usuario especifica qué información desea que se tenga en cuenta, "
                    "limita el plan a esas fuentes salvo que otra fuente sea claramente imprescindible. "

                    "8. Si una fuente no aporta información necesaria para responder, no la selecciones. "

                    "EJEMPLOS: "

                    "'¿Qué medicación toma el paciente X?' "
                    "-> required_data: ['get_medications']. "

                    "'¿Qué edad tiene el paciente X?' "
                    "-> required_data: ['get_patient']. "

                    "'¿Cuáles son los últimos resultados de laboratorio del paciente X?' "
                    "-> required_data: ['get_lab_results']. "

                    "'El paciente X presenta mareo. Teniendo en cuenta su medicación actual, "
                    "¿qué información adicional sería útil conocer?' "
                    "-> required_data: ['get_medications']. "

                    "'El paciente X presenta mareo. Teniendo en cuenta sus últimos análisis, "
                    "¿qué información adicional sería útil conocer?' "
                    "-> required_data: ['get_lab_results']. "

                    "'El paciente X presenta mareo y debilidad. Teniendo en cuenta su edad, "
                    "medicación y resultados de laboratorio, ¿qué información adicional sería útil conocer?' "
                    "-> required_data: ['get_patient', 'get_medications', 'get_lab_results']. "

                    "No recuperes datos tú mismo. "
                    "No inventes fuentes. "
                    "No expliques tu decisión. "
                    "Devuelve únicamente las fuentes necesarias mediante el formato JSON solicitado."
                )
            },
            {
                "role": "user",
                "content": state["question"]
            }
        ]
    )

    content = response.choices[0].message.content

    planner_output = json.loads(content)

    required_data = planner_output["required_data"]

    console.print(
        "[bold]Datos necesarios:[/bold]"
    )
    console.print(
        Pretty(required_data)
    )

    return {
        "required_data": required_data
    }


# ============================================================
# Validador
# ============================================================

def validator(state: RouterState):
    console.print()
    console.print(
        Panel(
            "[bold]Comprobando que se han recuperado todos los datos...[/bold]",
            title="🔍 VALIDATOR",
            border_style="blue"
        )
    )

    required_data = state["required_data"]
    patient_context = state["patient_context"]

    missing_data = [
        item
        for item in required_data
        if item not in patient_context
    ]

    validation_ok = len(missing_data) == 0

    if validation_ok:
        console.print(
            "[green]✓ Todos los datos requeridos han sido recuperados.[/green]"
        )
    else:
        console.print(
            "[red]✗ Faltan datos requeridos:[/red]"
        )
        console.print(
            Pretty(missing_data)
        )

    if validation_ok:
        retry_count = state["retry_count"]
    else:
        retry_count = state["retry_count"] + 1

    return {
        "validation_ok": validation_ok,
        "missing_data": missing_data,
        "retry_count": retry_count
    }


# ============================================================
# AGENTE DE SÍNTOMAS
# ============================================================


def symptom_agent(state: RouterState):

    console.print()
    console.print(
        Panel(
            "[bold]Analizando los síntomas descritos...[/bold]",
            title="🩺 AGENTE DE SÍNTOMAS",
            border_style="yellow"
        )
    )

    console.print(
        f"[dim]Intento del agente de síntomas: "
        f"{state['symptom_retry_count'] + 1}[/dim]"
    )

    response = groq_completion_with_retry(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un agente especializado exclusivamente en el análisis "
                    "de síntomas dentro de un sistema clínico educativo. "

                    "Tu tarea es identificar los síntomas mencionados por el usuario "
                    "y qué preguntas adicionales sería útil realizar sobre ellos. "

                    "NO debes proponer pruebas diagnósticas. "
                    "NO debes recomendar tratamientos ni cambios de medicación. "
                    "NO debes realizar un diagnóstico definitivo. "

                    "Tu respuesta será utilizada por otro agente del sistema. "

                    "Devuelve EXCLUSIVAMENTE un objeto JSON válido con esta estructura: "
                    "{"
                    '  "symptoms": ["síntoma 1", "síntoma 2"], '
                    '  "questions": ["pregunta 1", "pregunta 2"]'
                    "} "

                    "No escribas explicaciones antes ni después del JSON. "
                    "No utilices bloques de código Markdown."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Pregunta del usuario:\n"
                    f"{state['question']}\n\n"
                )
            }
        ]
    )

    analysis = response.choices[0].message.content


    try:
        analysis = json.loads(analysis)

    except json.JSONDecodeError:
        analysis = {}

    # if "symptoms" not in analysis:
    #     analysis["symptoms"] = []

    # if "questions" not in analysis:
    #     analysis["questions"] = []

    console.print()
    console.print(
        Panel(
            Pretty(analysis),
            title="🔎 Análisis de síntomas",
            border_style="yellow"
        )
    )

    return {
        "symptom_analysis": analysis
    }


def after_symptom_analysis(
    state: RouterState
) -> Literal["clinical", "no_symptoms"]:

    symptoms = state["symptom_analysis"].get("symptoms", [])

    if symptoms:
        return "clinical"

    return "no_symptoms"


def symptom_validator(state: RouterState):

    analysis = state["symptom_analysis"]

    validation_ok = (
        isinstance(analysis, dict)
        and isinstance(analysis.get("symptoms"), list)
        and isinstance(analysis.get("questions"), list)
    )

    if validation_ok:
        retry_count = state["symptom_retry_count"]
    else:
        retry_count = state["symptom_retry_count"] + 1

    return {
        "symptom_validation_ok": validation_ok,
        "symptom_retry_count": retry_count
    }


def after_symptom_validation(
    state: RouterState
) -> Literal["valid", "retry", "failed"]:

    if state["symptom_validation_ok"]:
        return "valid"

    if state["symptom_retry_count"] < 2:
        return "retry"

    return "failed"



def symptom_failure_handler(state: RouterState):

    return {
        "response": (
            "No se ha podido analizar correctamente la información "
            "sobre los síntomas."
        )
    }



CLINICAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "known_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string"
                    },
                    "source": {
                        "type": "string",
                        "enum": [
                            "user_question",
                            "get_patient",
                            "get_medications",
                            "get_lab_results"
                        ]
                    }
                },
                "required": [
                    "fact",
                    "source"
                ],
                "additionalProperties": False
            }
        },
        "clinical_reasoning": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string"
                    },
                    "based_on": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "user_question",
                                "get_patient",
                                "get_medications",
                                "get_lab_results"
                            ]
                        }
                    }
                },
                "required": [
                    "hypothesis",
                    "based_on"
                ],
                "additionalProperties": False
            }
        },
        "additional_information": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "response": {
            "type": "string"
        }
    },
    "required": [
        "known_facts",
        "clinical_reasoning",
        "additional_information",
        "response"
    ],
    "additionalProperties": False
}

# ============================================================
# AGENTE CLÍNICO
# ============================================================

def clinical_agent(state: RouterState):

    console.print()
    console.print(
        Panel(
            "[bold]Analizando la información clínica disponible...[/bold]",
            title="🩺 AGENTE CLÍNICO",
            border_style="magenta"
        )
    )

    # Mostramos el contexto del paciente
    if state["patient_context"]:
        console.print(
            Panel(
                Pretty(state["patient_context"]),
                title="📋 Contexto recibido del agente de datos",
                border_style="blue"
            )
        )
    else:
        console.print(
            "[dim]No se ha recibido contexto previo del agente de datos.[/dim]"
        )

    if state["symptom_analysis"]:
        console.print()
        console.print(
            Panel(
                Pretty(state["symptom_analysis"]),
                title="🔎 Análisis recibido del agente de síntomas",
                border_style="yellow"
            )
        )

    question = state["question"]
    patient_context = state["patient_context"]
    symptom_analysis = state["symptom_analysis"]

    symptoms = symptom_analysis.get("symptoms", [])
    questions = symptom_analysis.get("questions", [])
    semantic_issues = state["semantic_validation_issues"]
    structure_issues = state["clinical_validation_issues"]

    response = groq_completion_with_retry(
        model=CLINICAL_MODEL,
        max_completion_tokens=900,
        reasoning_effort="low",
        include_reasoning=False,
        fallback_to_json_object=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_analysis",
                "strict": True,
                "schema": CLINICAL_RESPONSE_SCHEMA
            }
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un agente de razonamiento clínico dentro de un sistema educativo de demostración. "

                    "Tu tarea es generar un análisis clínico prudente a partir de la pregunta del usuario, "
                    "los síntomas identificados y los datos recuperados. "
                    "Puedes utilizar conocimiento clínico general únicamente para formular hipótesis "
                    "o diagnósticos diferenciales, nunca para inventar hechos sobre el paciente. "
                    "No inventes datos, diagnósticos confirmados ni antecedentes. "
                    "No recomiendes tratamientos ni cambios de medicación. "

                    "Debes devolver EXCLUSIVAMENTE un objeto JSON válido con estos cuatro campos: "
                    "{"
                    '"known_facts": [{"fact": "...", "source": "..."}], '
                    '"clinical_reasoning": [{"hypothesis": "...", "based_on": ["..."]}], '
                    '"additional_information": ["..."], '
                    '"response": "..."'
                    "}. "

                    "FUENTES PERMITIDAS: "
                    "'user_question', 'get_patient', 'get_medications', 'get_lab_results'. "

                    "KNOWN_FACTS: "
                    "Incluye únicamente hechos explícitos presentes en la pregunta o en los datos recuperados. "
                    "Cada elemento debe contener exactamente 'fact' y 'source'. "
                    "Incluye como máximo 4 hechos relevantes. "

                    "Si source es 'user_question', conserva la información expresada por el usuario "
                    "sin intensificarla ni convertirla en términos clínicos diferentes. "
                    "Por ejemplo, 'mucha sed' debe mantenerse como 'mucha sed', "
                    "no como 'sed excesiva' ni 'polidipsia'. "

                    "Cuando source sea 'get_medications', copia EXACTAMENTE el texto del medicamento "
                    "tal como aparece en los datos recuperados. "
                    "No elimines, traduzcas, abrevies ni reformules ninguna parte del nombre, "
                    "dosis, presentación o duración. "

                    "Cuando source sea 'get_lab_results', copia EXACTAMENTE el nombre del laboratorio, "
                    "el valor y la unidad recibidos. "

                    "No incluyas el identificador técnico del paciente salvo que sea necesario "
                    "para responder a la consulta. "

                    "CLINICAL_REASONING: "
                    "Cada elemento debe contener exactamente 'hypothesis' y 'based_on'. "
                    "Incluye entre 1 y 2 hipótesis cuando exista información suficiente para formularlas. "
                    "Formula únicamente posibilidades clínicas, nunca diagnósticos confirmados. "

                    "based_on solo puede contener uno o varios de estos valores exactos: "
                    "'user_question', 'get_patient', 'get_medications', 'get_lab_results'. "
                    "'symptom_analysis' NO es una fuente válida. "
                    "Si utilizas un síntoma identificado por symptom_analysis, "
                    "utiliza 'user_question' como fuente porque el síntoma procede de la pregunta del usuario. "

                    "Cada fuente incluida en based_on debe apoyar realmente la hipótesis. "
                    "No incluyas get_lab_results si la hipótesis no depende de un resultado de laboratorio. "
                    "No incluyas get_medications si la hipótesis no depende de la medicación recuperada. "

                    "Puedes utilizar conocimiento clínico general para relacionar síntomas con posibles causas. "
                    "Ese conocimiento sirve para razonar, pero no constituye una nueva fuente de datos del paciente. "

                    "FIDELIDAD A LOS SÍNTOMAS: "
                    "No atribuyas al paciente síntomas, signos o manifestaciones que no aparezcan "
                    "en la pregunta o en symptom_analysis. "
                    "No sustituyas un síntoma por otro clínicamente diferente. "
                    "Por ejemplo, mareo no equivale a síncope, vértigo, desorientación "
                    "ni pérdida de conciencia. "

                    "MEDICACIÓN: "
                    "get_medications únicamente confirma qué medicamentos toma el paciente. "
                    "Puedes mencionarlos literalmente, pero no infieras a partir de ellos "
                    "diagnósticos, antecedentes o enfermedades. "
                    "No atribuyas a un medicamento una clase farmacológica, indicación, mecanismo de acción, "
                    "efecto adverso, efecto fisiológico o categoría terapéutica si esa información "
                    "no aparece explícitamente en los datos recuperados. "
                    "No utilices tu conocimiento farmacológico general para completar esas propiedades. "
                    "Sí puedes plantear de forma prudente que sería útil conocer si existe una relación temporal "
                    "entre la toma o un cambio de medicación y la aparición de los síntomas. "

                    "LABORATORIO: "
                    "Puedes mencionar literalmente los valores y unidades recuperados. "
                    "Si falta la unidad o el rango de referencia, no clasifiques el resultado como "
                    "alto, bajo, elevado, reducido, disminuido, normal, anormal, alterado o patológico. "
                    "Si reference_range es None, no utilices ese valor para apoyar una hipótesis "
                    "sobre una enfermedad o alteración. "
                    "Puedes indicar que no puede clasificarse con la información disponible. "

                    "ADDITIONAL_INFORMATION: "
                    "Incluye como máximo 4 elementos con información adicional concreta y útil "
                    "para valorar las posibles causas. "
                    "Puedes solicitar nuevas mediciones, antecedentes, características de los síntomas "
                    "o pruebas que serían útiles, pero exprésalas siempre como información que falta, "
                    "nunca como hechos ya conocidos. "
                    "Si get_lab_results no está disponible, no solicites rangos de referencia "
                    "de análisis inespecíficos que no hayan sido recuperados. "

                    "RESPONSE: "
                    "Responde directamente a la pregunta del usuario utilizando únicamente conclusiones "
                    "compatibles con las reglas anteriores. "
                    "No describas el funcionamiento interno del sistema ni utilices frases como "
                    "'se han recuperado datos' o 'se listan X medicamentos'. "
                    "Máximo 4 frases. "

                    "PRIORIDAD DE FORMATO: "
                    "Debes completar SIEMPRE los cuatro campos antes de finalizar la respuesta. "
                    "Es preferible devolver listas más cortas que omitir un campo. "
                    "NO finalices el JSON hasta haber escrito known_facts, clinical_reasoning, "
                    "additional_information y response. "
                    "Si necesitas reducir la longitud, reduce el número de elementos, "
                    "pero nunca elimines ninguno de los cuatro campos. "

                    "Los cuatro campos son obligatorios. "
                    "Si una lista no tiene elementos, devuelve []. "
                    "No escribas nada fuera del JSON y no utilices Markdown."
                )
            },
            {
                "role": "user",
                "content": f"""
            Pregunta:
            {question}

            Datos del paciente:
            {patient_context}

            Análisis de síntomas:
            {symptom_analysis}

            Errores semánticos detectados en un intento anterior:
            {semantic_issues}

            Errores de estructura detectados en un intento anterior:
            {structure_issues}

            Si alguna de estas listas contiene errores, debes corregirlos
            en la nueva respuesta.

            No reformules simplemente el mismo razonamiento problemático:
            elimínalo o modifícalo de forma que deje de cometer el error indicado.

                        Si un error semántico indica que has atribuido a un medicamento
            una propiedad farmacológica que no aparece en los datos recuperados,
            DEBES ELIMINAR completamente esa propiedad en el siguiente intento.

            No la reformules con palabras como "posible", "podría", "relacionado",
            "asociado" o expresiones equivalentes.

            Por ejemplo, si un intento anterior ha sido rechazado por afirmar
            que metoprolol produce hipotensión, NO vuelvas a escribir:
            "posible efecto hipotensivo del metoprolol",
            "el metoprolol podría bajar la presión",
            ni ninguna afirmación equivalente.

            En su lugar, limita el razonamiento a la información disponible.
            Puedes indicar que sería útil conocer la relación temporal entre
            la toma de la medicación y la aparición del síntoma, sin afirmar
            qué efecto produce el medicamento.

            Del mismo modo, si no se dispone de propiedades farmacológicas
            explícitas de la insulina, no afirmes que provoca hipoglucemia.
            Puedes solicitar valores de glucosa como información adicional
            si son clínicamente útiles, pero no atribuir su alteración
            directamente al medicamento.
            
            Si algún resultado tiene reference_range=None, NO vuelvas a describirlo
            como elevado, reducido, alto, bajo, normal, anormal, alterado o patológico.

            Por ejemplo, si la creatinina es 2.0223 mg/dL pero reference_range=None,
            NO escribas "creatinina elevada".

            Escribe en su lugar algo como:
            "Creatinina: 2.0223 mg/dL. No se clasifica como elevada o normal porque
            no se dispone del rango de referencia."

            Si un error semántico anterior señala una interpretación indebida de
            laboratorio, debes eliminar esa interpretación en el siguiente intento.

            En ese caso, limita el razonamiento a indicar que el resultado
            no puede interpretarse con la información disponible y añade
            la unidad, el rango de referencia o el contexto de la medición
            a additional_information.
            """
            }
        ]
    )

    raw_analysis = response.choices[0].message.content

    # 1. Vemos exactamente qué ha devuelto Groq
    console.print(
        Panel(
            raw_analysis,
            title="🔧 Respuesta cruda del agente clínico"
        )
    )

    # 2. Intentamos convertirlo a JSON
    try:
        clinical_analysis = json.loads(raw_analysis)
    except json.JSONDecodeError:
        clinical_analysis = {}

    # 3. Ahora sí mostramos el JSON procesado
    console.print(
        Panel(
            Pretty(clinical_analysis),
            title="🧠 Análisis clínico estructurado"
        )
    )

    # 4. Guardamos ambos resultados en el estado
    return {
        "clinical_analysis": clinical_analysis,
        "response": clinical_analysis.get(
            "response",
            "No se ha podido generar correctamente el análisis clínico."
        )
    }


def clinical_validator(state: RouterState):

    analysis = state["clinical_analysis"]

    issues = []

    if not isinstance(analysis, dict):
        issues.append("La respuesta clínica no es un objeto JSON.")

    else:
        known_facts = analysis.get("known_facts")

        if not isinstance(known_facts, list):
            issues.append("Falta known_facts o no es una lista.")

        else:
            allowed_sources = {
                "get_patient",
                "get_medications",
                "get_lab_results",
                "user_question"
            }

            for i, item in enumerate(known_facts):

                if not isinstance(item, dict):
                    issues.append(
                        f"known_facts[{i}] no es un objeto JSON."
                    )
                    continue

                if set(item.keys()) != {"fact", "source"}:
                    issues.append(
                        f"known_facts[{i}] debe contener exactamente fact y source."
                    )

                if not isinstance(item.get("fact"), str) or not item.get("fact", "").strip():
                    issues.append(
                        f"known_facts[{i}] tiene un fact inválido."
                    )

                if item.get("source") not in allowed_sources:
                    issues.append(
                        f"known_facts[{i}] tiene una fuente inválida."
                    )

        clinical_reasoning = analysis.get("clinical_reasoning")

        if not isinstance(clinical_reasoning, list):
            issues.append("Falta clinical_reasoning o no es una lista.")

        else:
            allowed_sources = {
                "get_patient",
                "get_medications",
                "get_lab_results",
                "user_question"
            }

            for i, item in enumerate(clinical_reasoning):

                if not isinstance(item, dict):
                    issues.append(
                        f"clinical_reasoning[{i}] no es un objeto JSON."
                    )
                    continue

                if set(item.keys()) != {"hypothesis", "based_on"}:
                    issues.append(
                        f"clinical_reasoning[{i}] debe contener exactamente "
                        f"hypothesis y based_on."
                    )

                if (
                    not isinstance(item.get("hypothesis"), str)
                    or not item.get("hypothesis", "").strip()
                ):
                    issues.append(
                        f"clinical_reasoning[{i}] tiene una hypothesis inválida."
                    )

                based_on = item.get("based_on")

                if not isinstance(based_on, list) or not based_on:
                    issues.append(
                        f"clinical_reasoning[{i}] tiene un based_on inválido."
                    )

                elif any(
                    source not in allowed_sources
                    for source in based_on
                ):
                    issues.append(
                        f"clinical_reasoning[{i}] contiene una fuente inválida."
                    )

        if not isinstance(analysis.get("additional_information"), list):
            issues.append("Falta additional_information o no es una lista.")

        if not isinstance(analysis.get("response"), str):
            issues.append("Falta response o no es un texto.")

    validation_ok = len(issues) == 0


    if validation_ok:
        retry_count = state["clinical_structure_retry_count"]
    else:
        retry_count = state["clinical_structure_retry_count"] + 1

    console.print()
    console.print(
        Panel(
            Pretty({
                "valid": validation_ok,
                "issues": issues,
                "retry_count": retry_count
            }),
            title="🔍 CLINICAL VALIDATOR",
            border_style="cyan"
        )
    )

    return {
        "clinical_validation_ok": validation_ok,
        "clinical_validation_issues": issues,
        "clinical_structure_retry_count": retry_count
    }



def validate_lab_units(
    clinical_analysis,
    patient_context
):
    issues = []

    lab_results = (
        patient_context
        .get("get_lab_results", {})
        .get("lab_results", {})
    )

    for lab_name, lab_data in lab_results.items():

        # Formato FHIR estructurado
        if isinstance(lab_data, dict):
            lab_value = lab_data.get("value")
            provided_unit = lab_data.get("unit")

        # Compatibilidad con mocks antiguos
        else:
            lab_value = lab_data
            provided_unit = None

        if (
            not isinstance(lab_value, (int, float))
            or isinstance(lab_value, bool)
        ):
            continue

        pattern = (
            rf"(?<![\d.,])"
            rf"{re.escape(str(lab_value))}"
            rf"\s*(mg/dL|mmol/L|g/L)"
            rf"(?![\w/])"
        )

        for field in [
            "known_facts",
            "clinical_reasoning",
            "additional_information",
            "response"
        ]:
            content = clinical_analysis.get(
                field,
                []
            )

            if isinstance(content, list):
                text = " ".join(
                    str(item)
                    for item in content
                )
            else:
                text = str(content)

            matches = re.findall(
                pattern,
                text,
                re.IGNORECASE
            )

            for detected_unit in matches:

                # No había unidad en la fuente
                if not provided_unit:
                    issues.append(
                        f"Se ha añadido una unidad no proporcionada "
                        f"al resultado '{lab_name}' ({lab_value}) "
                        f"en el campo '{field}'."
                    )
                    continue

                # La unidad escrita no coincide con FHIR
                if (
                    detected_unit.lower()
                    != provided_unit.lower()
                ):
                    issues.append(
                        f"Se ha utilizado la unidad "
                        f"'{detected_unit}' para '{lab_name}', "
                        f"pero la fuente proporciona "
                        f"'{provided_unit}' "
                        f"en el campo '{field}'."
                    )

    return list(dict.fromkeys(issues))

def validate_fact_sources(clinical_analysis, patient_context, question):
    issues = []

    known_facts = clinical_analysis.get("known_facts", [])

    patient_data = patient_context.get("get_patient", {})
    medications_data = patient_context.get("get_medications", {})
    lab_results = (
        patient_context
        .get("get_lab_results", {})
        .get("lab_results", {})
    )

    patient_age = patient_data.get("age")
    patient_name = patient_data.get("name")
    medications = medications_data.get("medications", [])

    for i, item in enumerate(known_facts):

        if not isinstance(item, dict):
            continue

        fact = item.get("fact", "")
        source = item.get("source")

        if not isinstance(fact, str):
            continue

        fact_lower = fact.lower()

        # =====================================================
        # GET_PATIENT
        # =====================================================

        if source == "get_patient":

            # Evitar que se atribuyan medicamentos a get_patient
            if any(
                medication.lower() in fact_lower
                for medication in medications
            ):
                issues.append(
                    f"known_facts[{i}] contiene información de medicación "
                    f"pero declara get_patient como fuente."
                )
                continue

            # Comprobar edad
            if re.search(
                r"\bedad\b|\baños?\b",
                fact,
                re.IGNORECASE
            ):
                numbers = re.findall(r"\b\d+\b", fact)

                if (
                    patient_age is not None
                    and numbers
                    and str(patient_age) not in numbers
                ):
                    issues.append(
                        f"known_facts[{i}] contiene una edad "
                        f"que no coincide con get_patient."
                    )

            # Comprobar nombre
            if re.search(
                r"\bnombre\b|\bse llama\b",
                fact,
                re.IGNORECASE
            ):
                if (
                    patient_name
                    and patient_name.lower() not in fact_lower
                ):
                    issues.append(
                        f"known_facts[{i}] contiene un nombre "
                        f"que no coincide con get_patient."
                    )

        # =====================================================
        # GET_MEDICATIONS
        # =====================================================

        elif source == "get_medications":

            if medications and not any(
                medication.lower() in fact_lower
                for medication in medications
            ):
                issues.append(
                    f"known_facts[{i}] no está respaldado "
                    f"por get_medications."
                )

        # =====================================================
        # GET_LAB_RESULTS
        # =====================================================

        elif source == "get_lab_results":

            lab_found = False

            # Alias básicos para permitir traducciones razonables
            # sin perder la comprobación del valor real.
            lab_aliases = {
                "glucose": ["glucose", "glucosa"],
                "creatinine": ["creatinine", "creatinina"],
                "calcium": ["calcium", "calcio"],
                "sodium": ["sodium", "sodio"],
                "potassium": ["potassium", "potasio"],
                "chloride": ["chloride", "cloruro"],
                "urea nitrogen": [
                    "urea nitrogen",
                    "nitrógeno ureico",
                    "nitrogeno ureico"
                ],
                "cholesterol": ["cholesterol", "colesterol"],
                "triglyceride": ["triglyceride", "triglicérido", "triglicerido"],
                "albumin": ["albumin", "albúmina", "albumina"],
                "globulin": ["globulin", "globulina"],
            }

            for lab_name, lab_data in lab_results.items():

                if isinstance(lab_data, dict):

                    lab_value = lab_data.get("value")

                    # Nombre corto:
                    # "Creatinine [Mass/volume] in Blood"
                    # -> "Creatinine"
                    short_name = (
                        lab_name.split("[", 1)[0]
                        .strip()
                        .lower()
                    )

                    aliases = [lab_name.lower(), short_name]

                    for key, extra_aliases in lab_aliases.items():
                        if key in lab_name.lower():
                            aliases.extend(extra_aliases)

                    mentions_lab = any(
                        alias in fact_lower
                        for alias in aliases
                    )

                else:
                    # Compatibilidad con mocks antiguos
                    lab_value = lab_data

                    aliases = [lab_name.lower()]

                    if lab_name.lower() == "glucose":
                        aliases.append("glucosa")

                    elif lab_name.lower() == "cholesterol":
                        aliases.append("colesterol")

                    mentions_lab = any(
                        alias in fact_lower
                        for alias in aliases
                    )

                if not mentions_lab:
                    continue

                lab_found = True

                # El valor sí debe coincidir EXACTAMENTE
                # con un valor recuperado.
                value_matches = (
                    lab_value is not None
                    and re.search(
                        rf"(?<![\d.,])"
                        rf"{re.escape(str(lab_value))}"
                        rf"(?!\d|[.,]\d)",
                        fact
                    )
                )

                if not value_matches:
                    issues.append(
                        f"known_facts[{i}] contiene un valor "
                        f"que no coincide con get_lab_results."
                    )

                break

            if not lab_found:
                issues.append(
                    f"known_facts[{i}] no está respaldado "
                    f"por get_lab_results."
                )
                
        elif source == "user_question":

            # Palabras poco informativas que no necesitamos
            # exigir literalmente en la pregunta
            ignored_words = {
                "el", "la", "los", "las",
                "un", "una", "unos", "unas",
                "de", "del", "al",
                "y", "o", "que",
                "paciente",
                "presenta", "presentado",
                "sintoma", "sintomas",
                "síntoma", "síntomas",
                "reportado", "reportados",
                "reportada", "reportadas", "id"
            }

            fact_words = {
                word
                for word in re.findall(
                    r"\b[\wáéíóúñ]+\b",
                    fact_lower
                )
                if word not in ignored_words
            }

            question_words = set(
                re.findall(
                    r"\b[\wáéíóúñ]+\b",
                    question.lower()
                )
            )

            unsupported_words = (
                fact_words - question_words
            )

            if unsupported_words:
                issues.append(
                    f"known_facts[{i}] contiene información "
                    f"que no aparece en user_question: "
                    f"{sorted(unsupported_words)}."
                )

        # =====================================================
        # FUENTE DESCONOCIDA
        # =====================================================

        else:
            issues.append(
                f"known_facts[{i}] utiliza una fuente "
                f"no permitida: '{source}'."
            )

    return list(dict.fromkeys(issues))



def validate_lab_interpretation(
    clinical_analysis,
    patient_context
):
    issues = []

    lab_results = (
        patient_context
        .get("get_lab_results", {})
        .get("lab_results", {})
    )

    # =====================================================
    # ¿EXISTE ALGÚN LABORATORIO INTERPRETABLE?
    # =====================================================

    has_interpretable_labs = any(
        isinstance(lab_data, dict)
        and lab_data.get("unit")
        and (
            lab_data.get("reference_range")
            or lab_data.get("range")
        )
        for lab_data in lab_results.values()
    )

    # Guardamos qué razonamientos ya han sido rechazados
    # para no generar después un segundo error redundante.
    invalid_lab_reasoning = set()

    for i, reasoning in enumerate(
        clinical_analysis.get(
            "clinical_reasoning",
            []
        )
    ):

        if not isinstance(reasoning, dict):
            continue

        based_on = reasoning.get(
            "based_on",
            []
        )

        if (
            "get_lab_results" in based_on
            and not has_interpretable_labs
        ):
            field = f"clinical_reasoning[{i}]"

            issues.append(
                f"{field} utiliza get_lab_results como apoyo "
                f"clínico, pero ninguno de los resultados "
                f"dispone de unidad y rango de referencia."
            )

            invalid_lab_reasoning.add(field)

    # =====================================================
    # TÉRMINOS QUE IMPLICAN INTERPRETACIÓN
    # =====================================================

    interpretation_pattern = (
        r"\b(?:"
        r"alto|alta|altos|altas|"
        r"bajo|baja|bajos|bajas|"
        r"elevado|elevada|elevados|elevadas|"
        r"disminuido|disminuida|"
        r"normal|normales|"
        r"anormal|anormales|"
        r"patológico|patológica|"
        r"hiperglucemia|hipoglucemia|"
        r"prediabetes|"
        r"control\s+gluc[eé]mico\s+sub[oó]ptimo|"
        r"alteraci[oó]n\s+gluc[eé]mica"
        r")\b"
    )

    # =====================================================
    # EXPRESIONES SEGURAS
    # =====================================================

    safe_patterns = [
        rf"\bno\s+puede\s+interpretarse[^.;]{{0,80}}{interpretation_pattern}",

        rf"\bno\s+se\s+puede\s+interpretar[^.;]{{0,80}}{interpretation_pattern}",

        rf"\bno\s+debe\s+clasificarse[^.;]{{0,80}}{interpretation_pattern}",

        rf"\bno\s+indica\s+por\s+s[ií]\s+mism[oa][^.;]{{0,80}}{interpretation_pattern}",

        rf"\b(?:se\s+)?requiere[^.;]{{0,100}}(?:unidad|rango)[^.;]{{0,100}}"
        rf"(?:determinar|saber)[^.;]{{0,50}}{interpretation_pattern}",

        rf"\b(?:se\s+)?necesita[^.;]{{0,100}}(?:unidad|rango)[^.;]{{0,100}}"
        rf"(?:determinar|saber)[^.;]{{0,50}}{interpretation_pattern}",

        rf"\bhace\s+falta[^.;]{{0,100}}(?:unidad|rango)[^.;]{{0,100}}"
        rf"(?:determinar|saber)[^.;]{{0,50}}{interpretation_pattern}",

        rf"^\s*(?:(?!{interpretation_pattern})[^.;]){{0,160}}"
        rf"\bno\s+se\s+puede\s+determinar\s+si\b"
        rf"[^.;]{{0,120}}{interpretation_pattern}"
        rf"(?:\s+o\s+no)?\s*$",

        rf"^\s*(?:(?!{interpretation_pattern})[^.;]){{0,180}}"
        rf"\bno\s+permite\s+concluir\s+si\b"
        rf"[^.;]{{0,120}}{interpretation_pattern}"
        rf"(?:[^.;]{{0,50}})?$",

        rf"^\s*(?:(?!{interpretation_pattern})[^.;]){{0,220}}"
        rf"\bno\s+se\s+puede\s+confirmar\s+ni\s+descartar\s+"
        rf"{interpretation_pattern}"
        rf"(?:[^.;]{{0,80}})?$",

        rf"^\s*(?:(?!{interpretation_pattern})[^.;]){{0,220}}"
        rf"\bno\s+se\s+puede\s+concluir\s+que\b"
        rf"[^.;]{{0,160}}{interpretation_pattern}"
        rf"(?:[^.;]{{0,80}})?$",
    ]

    # =====================================================
    # TEXTOS QUE VAMOS A REVISAR
    # =====================================================

    texts = []

    for i, item in enumerate(
        clinical_analysis.get(
            "clinical_reasoning",
            []
        )
    ):

        if not isinstance(item, dict):
            continue

        hypothesis = item.get(
            "hypothesis",
            ""
        )

        based_on = item.get(
            "based_on",
            []
        )

        texts.append(
            (
                f"clinical_reasoning[{i}]",
                hypothesis,
                based_on
            )
        )

    response = clinical_analysis.get(
        "response"
    )

    if isinstance(response, str):
        texts.append(
            (
                "response",
                response,
                []
            )
        )

    # =====================================================
    # REVISIÓN DE CADA LABORATORIO
    # =====================================================

    for lab_name, lab_data in lab_results.items():

        # FHIR estructurado
        if isinstance(lab_data, dict):

            lab_value = lab_data.get(
                "value"
            )

            unit = lab_data.get(
                "unit"
            )

            reference_range = (
                lab_data.get(
                    "reference_range"
                )
                or lab_data.get(
                    "range"
                )
            )

        # Compatibilidad con mocks antiguos
        else:

            lab_value = lab_data
            unit = None
            reference_range = None

        # Si tenemos unidad Y rango,
        # esta regla no interviene.
        if unit and reference_range:
            continue

        aliases = [
            lab_name
        ]

        # Compatibilidad con mocks antiguos
        if lab_name.lower() == "glucose":
            aliases.append(
                "glucosa"
            )

        elif lab_name.lower() == "cholesterol":
            aliases.append(
                "colesterol"
            )

        lab_pattern = "|".join(
            re.escape(alias)
            for alias in aliases
        )

        for field, text, based_on in texts:

            # Si ese clinical_reasoning ya fue rechazado
            # por utilizar laboratorios no interpretables,
            # no generamos otro error adicional.
            if field in invalid_lab_reasoning:
                continue

            if not isinstance(text, str):
                continue

            sentences = re.split(
                r"[.;]\s*",
                text
            )

            for sentence in sentences:

                mentions_lab = re.search(
                    rf"\b(?:{lab_pattern})\b",
                    sentence,
                    re.IGNORECASE
                )

                mentions_value = (
                    lab_value is not None
                    and re.search(
                        rf"(?<![\d.,])"
                        rf"{re.escape(str(lab_value))}"
                        rf"(?![\d.,])",
                        sentence
                    )
                )

                uses_lab_results = (
                    "get_lab_results"
                    in based_on
                )

                # -----------------------------------------
                # Interpretación explícita basada en labs
                # -----------------------------------------

                if (
                    uses_lab_results
                    and (mentions_lab or mentions_value)
                    and re.search(
                        interpretation_pattern,
                        sentence,
                        re.IGNORECASE
                    )
                ):

                    is_safe = any(
                        re.search(
                            pattern,
                            sentence,
                            re.IGNORECASE
                        )
                        for pattern in safe_patterns
                    )

                    if not is_safe:
                        issues.append(
                            "Se realiza una interpretación "
                            "clínica basada en "
                            "get_lab_results sin disponer "
                            "de unidad y rango de referencia "
                            f"en '{field}'."
                        )

                        continue

                # -----------------------------------------
                # No menciona este laboratorio
                # -----------------------------------------

                if not (
                    mentions_lab
                    or mentions_value
                ):
                    continue

                # -----------------------------------------
                # Lo menciona, pero sin interpretarlo
                # -----------------------------------------

                if not re.search(
                    interpretation_pattern,
                    sentence,
                    re.IGNORECASE
                ):
                    continue

                # -----------------------------------------
                # Interpretación explícitamente negada
                # -----------------------------------------

                if any(
                    re.search(
                        pattern,
                        sentence,
                        re.IGNORECASE
                    )
                    for pattern in safe_patterns
                ):
                    continue

                # -----------------------------------------
                # Interpretación indebida
                # -----------------------------------------

                issues.append(
                    f"Se interpreta el resultado "
                    f"'{lab_name}' sin disponer "
                    f"de unidad y rango de referencia "
                    f"en '{field}'."
                )

    return list(
        dict.fromkeys(
            issues
        )
    )


def validate_medication_diagnosis_inference(
    clinical_analysis,
    patient_context
):
    issues = []

    medications = (
        patient_context
        .get("get_medications", {})
        .get("medications", [])
    )

    inference_pattern = (
        r"\b(?:sugiere|indica|implica|confirma|demuestra)\b"
        r"[^.;]{0,160}"
        r"\b(?:"
        r"historia\s+de|"
        r"antecedentes?\s+de|"
        r"diagn[oó]stic\w*|"
        r"manejo\s+de|"
        r"tratad[oa]\s+(?:previamente\s+)?por|"
        r"en\s+tratamiento\s+por|"
        r"tratamiento\s+para"
        r")\b"
    )

    for i, item in enumerate(
        clinical_analysis.get("clinical_reasoning", [])
    ):
        if not isinstance(item, dict):
            continue

        hypothesis = item.get("hypothesis", "")
        based_on = item.get("based_on", [])

        if not isinstance(hypothesis, str):
            continue

        if "get_medications" not in based_on:
            continue

        mentions_medication = any(
            medication.lower() in hypothesis.lower()
            for medication in medications
        )

        if not mentions_medication:
            continue

        if re.search(
            inference_pattern,
            hypothesis,
            re.IGNORECASE
        ):
            issues.append(
                f"clinical_reasoning[{i}] infiere "
                f"un diagnóstico o antecedente "
                f"a partir de la medicación prescrita."
            )

    return list(dict.fromkeys(issues))


def validate_medication_properties(
    clinical_analysis,
    patient_context
):
    issues = []

    medications = (
        patient_context
        .get("get_medications", {})
        .get("medications", [])
    )

    # Si no se han recuperado medicamentos, esta validación no aplica.
    if not medications:
        return issues

    # Conceptos que implican atribuir propiedades farmacológicas.
    pharmacological_property_pattern = re.compile(
        r"\b(?:"
        r"antihipertensiv\w*|"
        r"betabloque\w*|"
        r"vasodilat\w*|"
        r"hipotensiv\w*|"
        r"hipoglucemi\w*|"
        r"hipoglucemiante\w*|"
        r"antiagreg\w*|"
        r"anticoagul\w*|"
        r"bradicardi\w*|"
        r"sistema\s+cardiovascular"
        r")\b",
        re.IGNORECASE
    )

    # ---------------------------------------------------------
    # 1. Comprobamos clinical_reasoning
    # ---------------------------------------------------------

    reasoning = clinical_analysis.get(
        "clinical_reasoning",
        []
    )

    for i, item in enumerate(reasoning):

        if not isinstance(item, dict):
            continue

        hypothesis = item.get("hypothesis")
        based_on = item.get("based_on", [])

        if not isinstance(hypothesis, str):
            continue

        # Solo nos interesa cuando utiliza la medicación
        # como fuente del razonamiento.
        if "get_medications" not in based_on:
            continue

        if re.search(
            pharmacological_property_pattern,
            hypothesis
        ):
            issues.append(
                f"clinical_reasoning[{i}] atribuye propiedades "
                f"farmacológicas a la medicación que no aparecen "
                f"en get_medications."
            )

    # ---------------------------------------------------------
    # 2. Comprobamos también la respuesta final
    # ---------------------------------------------------------

    response = clinical_analysis.get("response")

    if isinstance(response, str):

        medication_causal_pattern = re.compile(
            r"\b(?:"
            r"medicaci[oó]n|"
            r"medicaciones|"
            r"f[aá]rmaco|"
            r"f[aá]rmacos"
            r")\b"
            r"[^.;]{0,120}"
            r"\b(?:"
            r"afecta|afectan|"
            r"provoca|provocan|"
            r"causa|causan|"
            r"induce|inducen|"
            r"reduce|reducen|"
            r"disminuye|disminuyen"
            r")\b",
            re.IGNORECASE
        )

        if re.search(
            medication_causal_pattern,
            response
        ):
            issues.append(
                "response atribuye un efecto clínico a la medicación "
                "que no está explícitamente presente en get_medications."
            )

    return list(dict.fromkeys(issues))

def clinical_semantic_validator(state: RouterState):

    clinical_analysis = state["clinical_analysis"]
    patient_context = state["patient_context"]
    symptom_analysis = state["symptom_analysis"]
    question = state["question"]

    response = groq_completion_with_retry(
        model=GROQ_MODEL,
        max_completion_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un validador de seguridad y fidelidad clínica. "
                    "No realices un nuevo análisis clínico: evalúa únicamente si la respuesta "
                    "del agente está respaldada por la información disponible. "

                    "Marca la respuesta como inválida si detectas cualquiera de estos problemas: "
                    "hechos no respaldados por la pregunta o los datos recuperados; "
                    "valores, unidades o antecedentes inventados; "
                    "interpretaciones presentadas como hechos confirmados; "
                    "diagnósticos presentados como definitivos o confirmados sin estar explícitamente "
                    "recogidos en los datos disponibles; "
                    "recomendaciones de tratamiento o cambios de medicación; "
                    "diagnósticos inferidos únicamente a partir de la medicación prescrita. "

                    "Se permite formular hipótesis clínicas y diagnósticos diferenciales "
                    "cuando estén expresados como posibilidades y estén razonablemente apoyados "
                    "por la información disponible. "
                    "Expresiones como 'posible', 'podría', 'podría ser compatible con', "
                    "'sugiere' o 'una hipótesis sería' son válidas si la evidencia indicada "
                    "respalda esa posibilidad. "
                    "Una hipótesis clínica no debe rechazarse únicamente porque el diagnóstico "
                    "no aparezca explícitamente en los datos recuperados. "
                    "No debe considerarse válida si se presenta como una certeza, por ejemplo "
                    "'el paciente tiene X', 'el paciente padece X' o 'se confirma X', "
                    "sin que ese diagnóstico esté expresamente documentado. "

                    "Si un resultado de laboratorio no incluye unidad o rango de referencia, "
                    "no debe describirse como alto, bajo, normal, elevado, disminuido o patológico, "
                    "ni utilizarse ese valor aislado para justificar un diagnóstico. "

                    "REGLA ESTRICTA SOBRE MEDICAMENTOS: "

                    "Evalúa la fidelidad exclusivamente respecto a los datos proporcionados. "
                    "No utilices tu propio conocimiento farmacológico para considerar respaldada "
                    "una afirmación que no aparece en los datos recuperados. "

                    "Si DATOS RECUPERADOS únicamente contiene el nombre y la dosis de un medicamento, "
                    "solo está respaldado que el paciente toma ese medicamento y esa dosis. "

                    "NO están respaldadas, salvo que aparezcan explícitamente en DATOS RECUPERADOS, "
                    "afirmaciones sobre su clase farmacológica, indicación, mecanismo de acción, "
                    "efectos adversos, interacciones, efecto sobre la presión arterial, "
                    "efecto sobre la frecuencia cardíaca u otras propiedades clínicas. "

                    "Estas afirmaciones deben marcarse como inválidas incluso aunque sean "
                    "médicamente correctas según tu conocimiento general. "

                    "Por ejemplo, si los datos solo indican que el paciente toma metoprolol, "
                    "una afirmación como 'metoprolol es un antihipertensivo', "
                    "'metoprolol puede disminuir la presión arterial' o "
                    "'metoprolol puede causar mareo' NO está respaldada por los datos "
                    "y debe producir valid=false. "

                    "Sí está permitido afirmar literalmente que el paciente toma metoprolol "
                    "si ese medicamento aparece en DATOS RECUPERADOS. "
                    "También está permitido solicitar información adicional, por ejemplo "
                    "preguntar si el mareo aparece después de tomar la medicación, "
                    "sin afirmar que el medicamento sea la causa. "

                    "Devuelve EXCLUSIVAMENTE JSON válido con esta estructura: "
                    "{"
                    '"valid": true, '
                    '"issues": []'
                    "}. "

                    "Si existe algún problema, valid debe ser false e issues debe contener "
                    "una descripción concreta de cada error. "
                    "No utilices Markdown ni escribas nada fuera del JSON."

                    "Cuando una hipótesis o conclusión se apoye en get_lab_results, "
                    "comprueba que los resultados concretos utilizados realmente apoyan esa interpretación. "
                    "Si un resultado dispone de rango de referencia, la interpretación debe ser coherente "
                    "con ese rango. No utilices resultados dentro de rango como evidencia de una alteración "
                    "ni atribuyas a un resultado una función clínica que no esté justificada por los datos. "

                )
            },
            {
                "role": "user",
                "content": (
                    f"PREGUNTA ORIGINAL:\n{question}\n\n"
                    f"DATOS RECUPERADOS:\n{patient_context}\n\n"
                    f"ANÁLISIS DE SÍNTOMAS:\n{symptom_analysis}\n\n"
                    f"ANÁLISIS CLÍNICO A VALIDAR:\n{clinical_analysis}"
                )
            }
        ]
    )

    raw_validation = response.choices[0].message.content

    try:
        semantic_validation = json.loads(raw_validation)
    except json.JSONDecodeError:
        semantic_validation = {
            "valid": False,
            "issues": [
                "El validador semántico no devolvió JSON válido."
            ]
        }

    lab_unit_issues = validate_lab_units(
        clinical_analysis,
        patient_context
    )

    fact_source_issues = validate_fact_sources(
        clinical_analysis,
        patient_context,
        question
    )

    lab_interpretation_issues = validate_lab_interpretation(
        clinical_analysis,
        patient_context
    )

    medication_inference_issues = (
        validate_medication_diagnosis_inference(
            clinical_analysis,
            patient_context
        )
    )

    medication_property_issues = (
        validate_medication_properties(
            clinical_analysis,
            patient_context
        )
    )

    deterministic_issues = list(dict.fromkeys(
        lab_unit_issues
        + fact_source_issues
        + lab_interpretation_issues
        + medication_inference_issues
        + medication_property_issues
    ))

    semantic_validation["issues"] = (
        semantic_validation.get("issues", [])
        + deterministic_issues
    )

    semantic_validation["valid"] = (
        semantic_validation.get("valid") is True
        and len(deterministic_issues) == 0
    )

    console.print()
    console.print(
        Panel(
            Pretty(semantic_validation),
            title="🛡️ SEMANTIC VALIDATOR"
        )
    )

    if semantic_validation.get("valid", False):
        retry_count = state["clinical_retry_count"]
    else:
        retry_count = state["clinical_retry_count"] + 1

    return {
        "semantic_validation_ok": semantic_validation.get("valid", False),
        "semantic_validation_issues": semantic_validation.get("issues", []),
        "clinical_retry_count": retry_count
    }


def clinical_failed(state: RouterState):

    console.print()
    console.print(
        Panel(
            "[bold red]No se ha podido generar una respuesta clínica "
            "que supere las validaciones de seguridad.[/bold red]",
            title="⚠️ VALIDACIÓN CLÍNICA FALLIDA",
            border_style="red"
        )
    )

    return {
        "response": (
            "No se ha podido generar una respuesta clínica válida "
            "después de varios intentos."
        )
    }

def failure_handler(state: RouterState):

    console.print()
    console.print(
        Panel(
            (
                "[bold red]No se han podido recuperar "
                "todos los datos necesarios.[/bold red]\n\n"
                f"Datos pendientes: {state['missing_data']}"
            ),
            title="⚠️ ERROR DE RECUPERACIÓN",
            border_style="red"
        )
    )

    return {
        "response": (
            "No se han podido recuperar todos los datos necesarios "
            "del paciente después de varios intentos. "
            f"Datos pendientes: {state['missing_data']}"
        )
    }

# ============================================================
# DECISIONES DEL GRAFO
# ============================================================

def choose_route(
    state: RouterState
) -> Literal["datos", "clinico", "datos_y_clinico"]:

    return state["route"]


def after_data(
    state: RouterState
) -> Literal["end", "clinico"]:

    if state["route"] == "datos_y_clinico":
        return "clinico"

    return "end"


def after_validation(
    state: RouterState
) -> Literal["retry", "continue", "failed"]:

    if state["validation_ok"]:
        return "continue"

    if state["retry_count"] < 2:
        return "retry"

    return "failed"


def after_validation_router(state: RouterState):
    return {}

def after_symptom_validation_router(state: RouterState):
    return {}

def no_symptoms_handler(state: RouterState):

    return {
        "response": (
            "No se han identificado síntomas concretos en la consulta. "
            "Para realizar un análisis clínico es necesario proporcionar "
            "los síntomas o el motivo concreto de la consulta."
        )
    }


def after_semantic_validation(
    state: RouterState
) -> Literal["valid", "retry", "failed"]:

    if state["semantic_validation_ok"]:
        return "valid"

    if state["clinical_retry_count"] < 2:
        return "retry"

    return "failed"



def after_clinical_validation(
    state: RouterState
) -> Literal["continue", "retry", "failed"]:

    if state["clinical_validation_ok"]:
        return "continue"

    if state["clinical_structure_retry_count"] < 2:
        return "retry"

    return "failed"


# ============================================================
# CREACIÓN DEL GRAFO
# ============================================================

async def create_graph_app(client):

    # --------------------------------------------------------
    # HERRAMIENTAS MCP
    # --------------------------------------------------------

    mcp_tools = await client.list_tools()

    # --------------------------------------------------------
    # CONVERSIÓN MCP -> GROQ
    # --------------------------------------------------------

    groq_tools = []

    for tool in mcp_tools:

        groq_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        })

    # ========================================================
    # AGENTE DE DATOS
    # ========================================================

    async def data_agent(state: RouterState):

        if state["missing_data"]:
            data_to_retrieve = state["missing_data"]
        else:
            data_to_retrieve = state["required_data"]

        console.print()
        console.print(
            Panel(
                "[bold]Recuperando información del paciente...[/bold]",
                title="📂 AGENTE DE DATOS",
                border_style="green"
            )
        )

        console.print(
            "[dim]El agente recuperará los datos indicados por el Planner.[/dim]"
        )

        console.print(
            "[bold]Plan recibido:[/bold]"
        )

        console.print(
            Pretty(data_to_retrieve)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un agente especializado en recuperar datos de pacientes. "
                    "No conoces ningún dato del paciente por ti mismo. "
                    "Toda información del paciente debe obtenerse mediante "
                    "las herramientas disponibles. "
                    "Otro agente llamado Planner ya ha decidido qué fuentes "
                    "de datos son necesarias. "
                    f"Debes recuperar EXACTAMENTE estas fuentes: "
                    f"{data_to_retrieve}. "
                    "Utiliza las herramientas correspondientes a esas fuentes. "
                    "Debes ejecutar todas las herramientas indicadas en el plan. "
                    "No omitas ninguna herramienta indicada por el Planner. "
                    "No utilices herramientas que no aparezcan en el plan. "
                    "No inventes datos que no hayan sido devueltos por las herramientas. "
                    "Tu función es únicamente recuperar y presentar los datos solicitados. "
                    "No interpretes clínicamente los resultados obtenidos. "
                    "No determines si un resultado de laboratorio es normal, anormal, "
                    "alto, bajo, preocupante o compatible con una enfermedad. "
                    "No ofrezcas interpretar posteriormente resultados de laboratorio "
                    "si los datos recuperados no incluyen los rangos de referencia necesarios. "
                )
            },
            {
                "role": "user",
                "content": state["question"]
            }
        ]

        patient_data = dict(
            state["patient_context"]
        )

        while True:

            response = groq_completion_with_retry(
                model=GROQ_MODEL,
                messages=messages,
                tools=groq_tools
            )

            message = response.choices[0].message

            messages.append(
                message
            )

            # ------------------------------------------------
            # EL AGENTE HA TERMINADO
            # ------------------------------------------------

            if not message.tool_calls:

                console.print()
                console.print(
                    "[green]✓[/green] "
                    "[bold]Recuperación de datos completada[/bold]"
                )

                data_response = message.content
                if state["route"] == "datos":
                    data_sections = []

                    for source in state["required_data"]:
                        source_data = patient_data.get(source)
                        if not isinstance(source_data, dict):
                            continue

                        if source == "get_patient":
                            if "error" in source_data:
                                data_sections.append(
                                    f"Paciente: {source_data['error']}"
                                )
                                continue

                            patient_name = source_data.get("name")
                            patient_age = source_data.get("age")
                            if patient_name is not None and patient_age is not None:
                                data_sections.append(
                                    f"El paciente {patient_name} tiene {patient_age} años."
                                )
                            elif patient_name is not None:
                                data_sections.append(
                                    f"Paciente: {patient_name}."
                                )
                            elif patient_age is not None:
                                data_sections.append(
                                    f"Edad: {patient_age}."
                                )

                        elif source == "get_medications":
                            if "error" in source_data:
                                data_sections.append(
                                    f"Medicamentos: {source_data['error']}"
                                )
                                continue

                            medications = source_data.get("medications", [])
                            if isinstance(medications, list):
                                medication_lines = ["Medicamentos:"]
                                if medications:
                                    medication_lines.extend(
                                        f"- {medication}"
                                        for medication in medications
                                    )
                                else:
                                    medication_lines.append(
                                        "- No se devolvieron medicamentos."
                                    )
                                data_sections.append(
                                    "\n".join(medication_lines)
                                )

                        elif source == "get_lab_results":
                            if "error" in source_data:
                                data_sections.append(
                                    f"Resultados de laboratorio: {source_data['error']}"
                                )
                                continue

                            lab_results = source_data.get("lab_results", {})
                            if isinstance(lab_results, dict):
                                lab_lines = ["Resultados de laboratorio:"]
                                for name, result in lab_results.items():
                                    if not isinstance(result, dict):
                                        lab_lines.append(f"- {name}: {result}")
                                        continue

                                    value = result.get("value")
                                    value_text = (
                                        "null" if value is None else str(value)
                                    )
                                    unit = result.get("unit")
                                    result_text = f"- {name}: {value_text}"
                                    if unit is not None:
                                        result_text += f" {unit}"

                                    details = []
                                    if result.get("reference_range") is not None:
                                        details.append(
                                            "rango de referencia: "
                                            f"{result['reference_range']}"
                                        )
                                    if result.get("date") is not None:
                                        details.append(
                                            f"fecha: {result['date']}"
                                        )
                                    if details:
                                        result_text += (
                                            f" ({'; '.join(details)})"
                                        )
                                    lab_lines.append(result_text)

                                if not lab_results:
                                    lab_lines.append(
                                        "- No se devolvieron resultados de laboratorio."
                                    )
                                data_sections.append("\n".join(lab_lines))

                    data_response = "\n\n".join(data_sections)

                return {
                    "patient_context": patient_data,
                    "response": data_response
                }

            # ------------------------------------------------
            # EJECUTAMOS LAS TOOLS SOLICITADAS
            # ------------------------------------------------

            for tool_call in message.tool_calls:

                tool_name = (
                    tool_call.function.name
                )

                tool_arguments = json.loads(
                    tool_call.function.arguments
                )

                console.print()
                console.print(
                    Rule(
                        f"🔧 {tool_name}",
                        style="green"
                    )
                )

                console.print(
                    "[bold]Argumentos[/bold]"
                )

                console.print(
                    Pretty(tool_arguments)
                )

                # --------------------------------------------
                # LLAMADA MCP
                # --------------------------------------------

                result = await client.call_tool(
                    tool_name,
                    tool_arguments
                )

                console.print(
                    "[bold green]✓ Resultado MCP[/bold green]"
                )

                console.print(
                    Pretty(result.data)
                )

                patient_data[
                    tool_name
                ] = result.data

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result.data
                    )
                })

    # ========================================================
    # CONSTRUCCIÓN DEL GRAFO
    # ========================================================

    graph = StateGraph(
        RouterState
    )

    # --------------------------------------------------------
    # NODOS
    # --------------------------------------------------------

    graph.add_node(
        "router",
        router
    )

    graph.add_node(
        "planner",
        planner
    )

    graph.add_node(
        "data_agent",
        data_agent
    )

    graph.add_node(
        "validator",
        validator
    )

    graph.add_node(
        "failure_handler",
        failure_handler
    )

    graph.add_node(
        "symptom_agent",
        symptom_agent
    )

    graph.add_node(
        "no_symptoms_handler",
        no_symptoms_handler
    )

    graph.add_node(
        "clinical_agent",
        clinical_agent
    )

    graph.add_node(
        "clinical_validator",
        clinical_validator
    )

    graph.add_node(
        "clinical_semantic_validator",
        clinical_semantic_validator
    )

    graph.add_node(
        "after_validation_router",
        after_validation_router
    )

    graph.add_node(
        "symptom_validator",
        symptom_validator
    )

    graph.add_node(
        "symptom_failure_handler",
        symptom_failure_handler
    )

    graph.add_node(
        "after_symptom_validation_router",
        after_symptom_validation_router
    )

    graph.add_node(
        "clinical_failed",
        clinical_failed
    )

    # --------------------------------------------------------
    # INICIO
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "router"
    )

    # --------------------------------------------------------
    # ROUTER
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "router",
        choose_route,
        {
            "datos": "planner",
            "clinico": "symptom_agent",
            "datos_y_clinico": "planner"
        }
    )

    # --------------------------------------------------------
    # DATOS
    # --------------------------------------------------------

    graph.add_edge(
        "planner",
        "data_agent"
    )

    graph.add_edge(
        "data_agent",
        "validator"
    )

    graph.add_conditional_edges(
        "validator",
        after_validation,
        {
            "retry": "data_agent",
            "continue": "after_validation_router",
            "failed": "failure_handler"
        }
    )

    graph.add_conditional_edges(
        "after_validation_router",
        after_data,
        {
            "end": END,
            "clinico": "symptom_agent"
        }
    )

    # --------------------------------------------------------
    # SÍNTOMAS
    # --------------------------------------------------------

    graph.add_edge(
        "symptom_agent",
        "symptom_validator"
    )

    graph.add_conditional_edges(
        "symptom_validator",
        after_symptom_validation,
        {
            "valid": "after_symptom_validation_router",
            "retry": "symptom_agent",
            "failed": "symptom_failure_handler"
        }
    )

    graph.add_conditional_edges(
        "after_symptom_validation_router",
        after_symptom_analysis,
        {
            "clinical": "clinical_agent",
            "no_symptoms": "no_symptoms_handler"
        }
    )

    # --------------------------------------------------------
    # CLÍNICO
    # --------------------------------------------------------

    graph.add_edge(
        "clinical_agent",
        "clinical_validator"
    )

    graph.add_conditional_edges(
        "clinical_validator",
        after_clinical_validation,
        {
            "continue": "clinical_semantic_validator",
            "retry": "clinical_agent",
            "failed": "clinical_failed"
        }
    )

    graph.add_conditional_edges(
        "clinical_semantic_validator",
        after_semantic_validation,
        {
            "valid": END,
            "retry": "clinical_agent",
            "failed": "clinical_failed"
        }
    )

    # --------------------------------------------------------
    # FINALES
    # --------------------------------------------------------

    graph.add_edge(
        "clinical_failed",
        END
    )

    graph.add_edge(
        "failure_handler",
        END
    )

    graph.add_edge(
        "no_symptoms_handler",
        END
    )

    # --------------------------------------------------------
    # COMPILACIÓN
    # --------------------------------------------------------

    app = graph.compile()

    return app, mcp_tools


# ============================================================
# EJECUTAR UNA CONSULTA
# ============================================================

async def execute_question(
    app,
    question: str,
    last_patient_id: str | None = None
):

    # --------------------------------------------------------
    # IDENTIFICADOR DEL PACIENTE
    # --------------------------------------------------------

    patient_id_match = re.search(
        r"\b[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}\b",
        question
    )

    if patient_id_match:

        last_patient_id = (
            patient_id_match.group(0)
        )

    elif last_patient_id:

        question = (
            f"{question}\n"
            f"El paciente al que se refiere la conversación "
            f"tiene el identificador {last_patient_id}."
        )

    # --------------------------------------------------------
    # ESTADO INICIAL
    # --------------------------------------------------------

    initial_state = {
        "question": question,
        "route": "",
        "required_data": [],
        "patient_context": {},
        "missing_data": [],
        "validation_ok": False,
        "retry_count": 0,
        "symptom_analysis": {},
        "symptom_validation_ok": False,
        "symptom_retry_count": 0,
        "clinical_analysis": {},
        "clinical_validation_ok": False,
        "clinical_validation_issues": [],
        "clinical_retry_count": 0,
        "semantic_validation_ok": False,
        "semantic_validation_issues": [],
        "clinical_structure_retry_count": 0,
        "response": ""
    }

    # --------------------------------------------------------
    # EJECUTAMOS EL GRAFO
    # --------------------------------------------------------

    final_state = await app.ainvoke(
        initial_state
    )

    return (
        final_state,
        last_patient_id
    )


# ============================================================
# MAIN - MODO CONSOLA
# ============================================================

async def main():

    # --------------------------------------------------------
    # CABECERA
    # --------------------------------------------------------

    console.print()

    console.print(
        Panel.fit(
            "[bold cyan]ASISTENTE CLÍNICO MULTIAGENTE[/bold cyan]\n"
            "[dim]LangGraph + MCP + Groq[/dim]",
            border_style="cyan"
        )
    )

    # --------------------------------------------------------
    # CONEXIÓN MCP
    # --------------------------------------------------------

    transport = StdioTransport(
        command=sys.executable,
        args=["mcp_server.py"],
        log_file=Path(
            "mcp_server_log.txt"
        )
    )

    client = Client(
        transport
    )

    async with client:

        # ----------------------------------------------------
        # CREAMOS EL GRAFO
        # ----------------------------------------------------

        app, mcp_tools = (
            await create_graph_app(
                client
            )
        )

        # ----------------------------------------------------
        # MOSTRAMOS LAS TOOLS DISPONIBLES
        # ----------------------------------------------------

        tools_table = Table(
            title="🔌 Herramientas MCP disponibles",
            show_header=True
        )

        tools_table.add_column(
            "Tool",
            style="cyan"
        )

        tools_table.add_column(
            "Descripción"
        )

        for tool in mcp_tools:

            tools_table.add_row(
                tool.name,
                tool.description
                or "Sin descripción"
            )

        console.print()
        console.print(
            tools_table
        )

        # ----------------------------------------------------
        # CONTEXTO DE CONVERSACIÓN
        # ----------------------------------------------------

        last_patient_id = None

        # ----------------------------------------------------
        # BUCLE DE CONSULTAS
        # ----------------------------------------------------

        while True:

            console.print()

            question = input(
                "Escribe tu consulta "
                "(o 'salir' para terminar): "
            ).strip()

            if question.lower() in {
                "salir",
                "exit",
                "quit"
            }:

                console.print(
                    "\n[cyan]Asistente finalizado.[/cyan]"
                )

                break

            if not question:
                continue

            # ------------------------------------------------
            # MOSTRAMOS PREGUNTA
            # ------------------------------------------------

            console.print()
            console.print(
                Panel(
                    question,
                    title="💬 PREGUNTA",
                    border_style="yellow"
                )
            )

            # ------------------------------------------------
            # EJECUTAMOS CONSULTA
            # ------------------------------------------------

            final_state, last_patient_id = (
                await execute_question(
                    app,
                    question,
                    last_patient_id
                )
            )

            # ------------------------------------------------
            # RESPUESTA FINAL
            # ------------------------------------------------

            console.print()
            console.print(
                Panel(
                    Markdown(
                        final_state[
                            "response"
                        ]
                    ),
                    title="💬 RESPUESTA FINAL",
                    border_style="green"
                )
            )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())