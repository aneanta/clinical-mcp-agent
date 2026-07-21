from dotenv import load_dotenv
load_dotenv()
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage

# Importa tu agente de LangGraph
from app.agent.graph import clinical_agent

app = FastAPI(title="TFM Clinical Agent API")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_session"


# --- FUNCIONES AUXILIARES PARA MANEJAR TIPOS DE MENSAJES ---

def get_message_content_str(msg) -> str:
    """Extrae el texto plano de un mensaje (sea objeto LangChain o dict, string o lista)."""
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Si el modelo devuelve bloques de contenido tipo [{'type': 'text', 'text': '...'}]
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)
    return str(content)


def is_human(msg) -> bool:
    """Verifica si un mensaje proviene del usuario."""
    if isinstance(msg, HumanMessage):
        return True
    if isinstance(msg, dict):
        role = msg.get("role") or msg.get("type")
        return role in ["user", "human"]
    return getattr(msg, "type", "") in ["user", "human"]


def check_tool_documents(msg):
    """Verifica si el mensaje es de una herramienta de documentos y extrae la lista."""
    is_tool = False
    name = ""
    content = None

    if isinstance(msg, ToolMessage):
        is_tool = True
        name = getattr(msg, "name", "") or ""
        content = msg.content
    elif isinstance(msg, dict) and (msg.get("role") == "tool" or msg.get("type") == "tool"):
        is_tool = True
        name = msg.get("name", "") or ""
        content = msg.get("content")

    if is_tool and "document" in name.lower():
        if isinstance(content, list):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return []
    return None


# --- ENDPOINT PRINCIPAL ---

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": request.thread_id}}

        # Invocación del agente
        result = clinical_agent.invoke(inputs, config=config)

        all_messages = result.get("messages", [])
        if not all_messages:
            return {"response": "No se recibió respuesta del agente.", "documents": []}

        # 1. Obtener la respuesta de texto del último mensaje
        last_message = all_messages[-1]
        response_text = get_message_content_str(last_message)

        # 2. Encontrar la posición del ÚLTIMO mensaje enviado por el usuario
        last_human_idx = -1
        for idx in range(len(all_messages) - 1, -1, -1):
            if is_human(all_messages[idx]):
                last_human_idx = idx
                break

        # 3. Buscar documentos SOLO si la herramienta devolvió archivos reales en ESTE turno
        current_documents = []
        if last_human_idx != -1:
            messages_in_this_turn = all_messages[last_human_idx + 1:]
            for msg in messages_in_this_turn:
                docs = check_tool_documents(msg)
                # Solo guardamos si docs es una lista y TIENE al menos 1 elemento
                if docs and isinstance(docs, list) and len(docs) > 0:
                    current_documents = docs

        return {
            "response": response_text,
            "documents": current_documents
        }

    except Exception as e:
        # Imprime la traza completa en la consola/terminal del backend para depuración
        print("\n❌ EXCEPCIÓN EN /api/chat:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")