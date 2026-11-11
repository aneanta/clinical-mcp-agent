import asyncio
import sys

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from multiagent_mcp import (
    create_graph_app,
    execute_question
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODELOS DE ENTRADA / SALIDA
# ============================================================

class ChatRequest(BaseModel):
    message: str
    last_patient_id: str | None = None


class ChatResponse(BaseModel):
    response: str

    route: str | None = None
    required_data: list[str] = []

    patient_context: dict = {}
    symptom_analysis: dict = {}
    clinical_analysis: dict = {}

    last_patient_id: str | None = None


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # --------------------------------------------------------
    # CREAMOS CONEXIÓN MCP
    # --------------------------------------------------------

    transport = StdioTransport(
        command=sys.executable,
        args=[
            str(
                BASE_DIR
                / "mcp_server.py"
            )
        ],
        log_file=BASE_DIR / "mcp_server_log.txt"
    )

    client = Client(
        transport
    )

    # --------------------------------------------------------
    # ABRIMOS MCP
    # --------------------------------------------------------

    async with client:

        # ----------------------------------------------------
        # CREAMOS EL GRAFO UNA SOLA VEZ
        # ----------------------------------------------------

        graph_app, mcp_tools = (
            await create_graph_app(
                client
            )
        )

        # ----------------------------------------------------
        # GUARDAMOS RECURSOS COMPARTIDOS
        # ----------------------------------------------------

        app.state.graph = graph_app
        app.state.mcp_client = client

        app.state.mcp_tools = [
            tool.name
            for tool in mcp_tools
        ]

        # Evitamos que dos consultas utilicen simultáneamente
        # la misma conexión stdio MCP.
        app.state.chat_lock = (
            asyncio.Lock()
        )

        print(
            "API iniciada correctamente."
        )

        print(
            "Herramientas MCP disponibles:",
            app.state.mcp_tools
        )

        # ----------------------------------------------------
        # FASTAPI EMPIEZA A RECIBIR PETICIONES
        # ----------------------------------------------------

        yield

    # --------------------------------------------------------
    # AL APAGAR FASTAPI, MCP SE CIERRA AUTOMÁTICAMENTE
    # --------------------------------------------------------

    print(
        "API finalizada."
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Asistente Clínico Multiagente",
    description=(
        "API del asistente clínico basado "
        "en LangGraph, MCP y Groq."
    ),
    version="0.1.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ENDPOINT RAÍZ
# ============================================================

@app.get("/")
async def root():

    return {
        "message": (
            "Asistente Clínico "
            "Multiagente API"
        ),
        "status": "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health(
    request: Request
):

    return {
        "status": "ok",
        "mcp_tools": (
            request.app.state.mcp_tools
        )
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(
    payload: ChatRequest,
    request: Request
):

    if not payload.message.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "El mensaje no puede "
                "estar vacío."
            )
        )

    try:

        # ----------------------------------------------------
        # RECUPERAMOS EL GRAFO
        # ----------------------------------------------------

        graph_app = (
            request.app.state.graph
        )

        # ----------------------------------------------------
        # EJECUTAMOS CONSULTA
        # ----------------------------------------------------

        async with (
            request.app.state.chat_lock
        ):

            (
                final_state,
                last_patient_id
            ) = await execute_question(
                graph_app,
                payload.message,
                payload.last_patient_id
            )

        # ----------------------------------------------------
        # RESPUESTA HTTP
        # ----------------------------------------------------

        return ChatResponse(

            response=final_state.get(
                "response",
                ""
            ),

            route=final_state.get(
                "route"
            ),

            required_data=final_state.get(
                "required_data",
                []
            ),

            patient_context=final_state.get(
                "patient_context",
                {}
            ),

            symptom_analysis=final_state.get(
                "symptom_analysis",
                {}
            ),

            clinical_analysis=final_state.get(
                "clinical_analysis",
                {}
            ),

            last_patient_id=(
                last_patient_id
            )
        )

    except Exception as exc:

        print(
            "Error procesando /chat:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Error procesando "
                "la consulta."
            )
        ) from exc