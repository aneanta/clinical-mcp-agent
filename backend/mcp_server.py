from fastmcp import FastMCP

from fhir_repository import (
    get_patient as fhir_get_patient,
    get_medications as fhir_get_medications,
    get_lab_results as fhir_get_lab_results,
)


mcp = FastMCP("Servidor de pacientes")


@mcp.tool
def get_patient(patient_id: str) -> dict:
    """Obtiene los datos generales de un paciente."""

    result = fhir_get_patient(patient_id)

    if "error" in result:
        return result

    # Mantenemos exactamente la interfaz
    # que ya esperaba el sistema.
    return {
        "name": result["name"],
        "age": result["age"]
    }


@mcp.tool
def get_medications(patient_id: str) -> dict:
    """Obtiene la medicación actual de un paciente."""

    return fhir_get_medications(patient_id)


@mcp.tool
def get_lab_results(patient_id: str) -> dict:
    """Obtiene los últimos resultados de laboratorio de un paciente."""

    return fhir_get_lab_results(patient_id)


if __name__ == "__main__":
    mcp.run()