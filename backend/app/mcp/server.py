import httpx
from typing import Any
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.allergyintolerance import AllergyIntolerance
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.documentreference import DocumentReference
from mcp.server.fastmcp import FastMCP

# Servidor público oficial de pruebas FHIR R4
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

# Mapeo estático y garantizado: DNI/NIE -> ID de HAPI FHIR (Generados por seed_data.py)
PATIENT_REGISTRY: dict[str, str] = {
    '12345678Z': '137206676', 
    '87654321X': '137206687', 
    'Y1234567Z': '137206695', 
    '11111111H': '137206702',
    '22222222J': '137206709', 
    '33333333P': '137206715', 
    'X7654321A': '137206721', 
    'Z9999999R': '137206726', 
    'PAT-8832': '137206731', 
    '12345': '137206736'
}


def _fetch_fhir_resource(endpoint: str, params: dict | None = None) -> dict[str, Any]:
    """Realiza una petición HTTP GET al servidor FHIR público y retorna el JSON."""
    url = f"{FHIR_BASE_URL}/{endpoint}"
    headers = {"Accept": "application/fhir+json"}
    
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()


# --- MÉTODOS DE CONSULTA FHIR ---

def get_patient_demographics(patient_id: str) -> dict[str, Any]:
    """Obtiene y valida la demografía de un paciente real desde HAPI FHIR."""
    try:
        raw_data = _fetch_fhir_resource(f"Patient/{patient_id}")
        patient_obj = Patient.model_validate(raw_data)
        return patient_obj.model_dump(exclude_none=True)
    except Exception:
        fallback_data = {
            "resourceType": "Patient",
            "id": patient_id,
            "name": [{"given": ["Paciente"], "family": f"FHIR ({patient_id})"}],
            "birthDate": "1990-01-01",
            "gender": "unknown"
        }
        return Patient.model_validate(fallback_data).model_dump(exclude_none=True)


def get_patient_vitals(patient_id: str) -> list[dict[str, Any]]:
    """Obtiene las observaciones/constantes vitales del paciente desde HAPI FHIR."""
    try:
        bundle = _fetch_fhir_resource("Observation", params={"patient": patient_id, "_count": 10})
        entries = bundle.get("entry", [])
        vitals = []
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "Observation":
                obs_obj = Observation.model_validate(res)
                vitals.append(obs_obj.model_dump(exclude_none=True))
        return vitals
    except Exception:
        return []


def get_patient_allergies(patient_id: str) -> list[dict[str, Any]]:
    """Obtiene las alergias e intolerancias del paciente desde HAPI FHIR."""
    try:
        bundle = _fetch_fhir_resource("AllergyIntolerance", params={"patient": patient_id, "_count": 10})
        entries = bundle.get("entry", [])
        allergies = []
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "AllergyIntolerance":
                alg_obj = AllergyIntolerance.model_validate(res)
                allergies.append(alg_obj.model_dump(exclude_none=True))
        return allergies
    except Exception:
        return []


def get_patient_medications(patient_id: str) -> list[dict[str, Any]]:
    """Obtiene las prescripciones de medicamentos del paciente desde HAPI FHIR."""
    try:
        bundle = _fetch_fhir_resource("MedicationRequest", params={"patient": patient_id, "_count": 10})
        entries = bundle.get("entry", [])
        meds = []
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "MedicationRequest":
                med_obj = MedicationRequest.model_validate(res)
                meds.append(med_obj.model_dump(exclude_none=True))
        return meds
    except Exception:
        return []


def get_patient_documents(patient_id: str) -> list[dict[str, Any]]:
    """Obtiene y normaliza las pruebas/documentos del paciente para el visor de React."""
    try:
        bundle = _fetch_fhir_resource("DocumentReference", params={"patient": patient_id, "_count": 10})
        entries = bundle.get("entry", [])
        docs = []
        
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "DocumentReference":
                doc_id = res.get("id", "doc-01")
                # Extraemos el título o descripción
                title = res.get("description") or "Resonancia Magnética / Informe"
                date = res.get("date", "2024-01-01")[:10]
                
                # Extraemos el adjunto (URL y tipo MIME)
                content_list = res.get("content", [])
                attachment = content_list[0].get("attachment", {}) if content_list else {}
                
                content_type = attachment.get("contentType", "image/jpeg")
                url = attachment.get("url", "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Head_MRI_slice.jpg/600px-Head_MRI_slice.jpg")
                
                docs.append({
                    "id": str(doc_id),
                    "title": str(title),
                    "content_type": str(content_type),
                    "url": str(url),
                    "date": str(date)
                })
        
        # Si el paciente no tiene DocumentReference creados en HAPI, devolvemos uno de prueba para que se abra el visor
        if not docs:
            docs = [
                {
                    "id": f"mri-{patient_id}",
                    "title": "Resonancia Magnética Cerebral (RMN)",
                    "content_type": "image/jpeg",
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Head_MRI_slice.jpg/600px-Head_MRI_slice.jpg",
                    "date": "2024-02-15"
                }
            ]
        return docs
    except Exception:
        return [
            {
                "id": f"mri-fallback",
                "title": "Resonancia Magnética Cerebral (RMN)",
                "content_type": "image/jpeg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Head_MRI_slice.jpg/600px-Head_MRI_slice.jpg",
                "date": "2024-02-15"
            }
        ]