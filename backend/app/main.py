from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agent.graph import clinical_agent
from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.graph import clinical_agent
from langchain_core.messages import HumanMessage
app = FastAPI(
    title="Clinical MCP Agent API",
    version="1.0.0",
    description="Backend para el asistente clínico basado en LangGraph y FastMCP"
)

# Permitir peticiones HTTP desde el Frontend (React/Next.js en http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "online", "service": "Clinical MCP Agent API"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Usamos un thread_id fijo para que LangGraph recuerde el DNI entre mensajes
    config = {"configurable": {"thread_id": "session_1"}}
    
    inputs = {"messages": [HumanMessage(content=request.message)]}
    result = clinical_agent.invoke(inputs, config=config)
    
    # Extraemos el objeto del último mensaje
    last_msg_obj = result["messages"][-1]
    
    # Leemos los documentos adjuntos de additional_kwargs (si los hay)
    documents = last_msg_obj.additional_kwargs.get("documents", [])
    
    # Devolvemos tanto el texto como la lista de documentos a React
    return {
        "text": last_msg_obj.content,
        "response": last_msg_obj.content,
        "documents": documents
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)