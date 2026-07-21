import httpx
import uuid
from typing import Any

FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

# 10 Perfiles Clínicos Completos con DNIs/NIEs válidos y Documentos de prueba
PATIENTS_TO_CREATE = [
    {
        "dni": "12345678Z",
        "given": "María", "family": "García López", "birth": "1985-04-12", "gender": "female",
        "vitals": [
            {"code": "8480-6", "text": "Tensión Arterial Sistólica", "value": 128, "unit": "mmHg"},
            {"code": "8462-4", "text": "Tensión Arterial Diastólica", "value": 82, "unit": "mmHg"},
            {"code": "8867-4", "text": "Frecuencia Cardíaca", "value": 72, "unit": "bpm"},
            {"code": "2339-0", "text": "Glucosa en sangre", "value": 95, "unit": "mg/dL"}
        ],
        "allergies": [
            {"text": "Penicilina", "criticality": "high"},
            {"text": "Polen de Olivo", "criticality": "low"}
        ],
        "meds": [
            {"text": "Ibuprofeno 600mg", "dosage": "1 comprimido cada 8 horas en caso de dolor"},
            {"text": "Antihistamínico Ebastina 20mg", "dosage": "1 comprimido cada 24 horas por la noche"}
        ],
        "documents": [
            {
                "title": "Resonancia Magnética Cerebral",
                "date": "2026-06-15",
                "contentType": "image/jpeg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/T1t2PD.jpg/800px-T1t2PD.jpg"
            },
            {
                "title": "Analítica de Sangre Completa",
                "date": "2026-01-10",
                "contentType": "application/pdf",
                "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            }
        ]
    },
    {
        "dni": "87654321X",
        "given": "Carlos", "family": "Rodríguez Martín", "birth": "1962-11-23", "gender": "male",
        "vitals": [
            {"code": "8480-6", "text": "Tensión Arterial Sistólica", "value": 148, "unit": "mmHg"},
            {"code": "8462-4", "text": "Tensión Arterial Diastólica", "value": 95, "unit": "mmHg"},
            {"code": "2339-0", "text": "Glucosa en sangre", "value": 165, "unit": "mg/dL"}
        ],
        "allergies": [
            {"text": "Aspirina (Ácido Acetilsalicílico)", "criticality": "high"}
        ],
        "meds": [
            {"text": "Enalapril 20mg", "dosage": "1 comprimido al día por la mañana"},
            {"text": "Metformina 850mg", "dosage": "1 comprimido con cada comida principal"}
        ],
        "documents": [
            {
                "title": "Radiografía de Tórax",
                "date": "2025-11-20",
                "contentType": "image/jpeg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/b/b3/Chest_Xray_PA_3-8-2010.png"
            }
        ]
    },
    {
        "dni": "Y1234567Z",
        "given": "Elena", "family": "Fernández Gómez", "birth": "1998-07-05", "gender": "female",
        "vitals": [
            {"code": "8867-4", "text": "Frecuencia Cardíaca", "value": 88, "unit": "bpm"},
            {"code": "8310-5", "text": "Temperatura Corporal", "value": 38.2, "unit": "°C"}
        ],
        "allergies": [
            {"text": "Frutos Secos (Nueces/Cacahuetes)", "criticality": "high"},
            {"text": "Látex", "criticality": "low"}
        ],
        "meds": [
            {"text": "Paracetamol 1g", "dosage": "1 comprimido cada 8 horas mientras persista la fiebre"},
            {"text": "Inhalador Ventolin (Salbutamol)", "dosage": "2 pulsaciones en caso de disnea"}
        ]
    },
    {
        "dni": "11111111H",
        "given": "Antonio", "family": "Navarro Sánchez", "birth": "1950-02-18", "gender": "male",
        "vitals": [
            {"code": "8480-6", "text": "Tensión Arterial Sistólica", "value": 135, "unit": "mmHg"},
            {"code": "8867-4", "text": "Frecuencia Cardíaca", "value": 64, "unit": "bpm"},
            {"code": "2708-6", "text": "Saturación de Oxígeno (SpO2)", "value": 94, "unit": "%"}
        ],
        "allergies": [
            {"text": "Polvo y Ácaros", "criticality": "low"}
        ],
        "meds": [
            {"text": "Atorvastatina 40mg", "dosage": "1 comprimido antes de acostarse"},
            {"text": "Sintrom (Acenocumarol) 4mg", "dosage": "Según pauta hematológica diaria"}
        ]
    },
    {
        "dni": "22222222J",
        "given": "Laura", "family": "Sora Pérez", "birth": "2001-09-30", "gender": "female",
        "vitals": [
            {"code": "8480-6", "text": "Tensión Arterial Sistólica", "value": 110, "unit": "mmHg"},
            {"code": "8462-4", "text": "Tensión Arterial Diastólica", "value": 70, "unit": "mmHg"},
            {"code": "8867-4", "text": "Frecuencia Cardíaca", "value": 65, "unit": "bpm"}
        ],
        "allergies": [
            {"text": "Amoxicilina", "criticality": "high"}
        ],
        "meds": [
            {"text": "Anticonceptivo Oral (Etinilestradiol)", "dosage": "1 comprimido diario"}
        ]
    },
    {
        "dni": "33333333P",
        "given": "Javier", "family": "López Ruiz", "birth": "1977-12-11", "gender": "male",
        "vitals": [
            {"code": "2339-0", "text": "Glucosa en sangre", "value": 210, "unit": "mg/dL"},
            {"code": "8480-6", "text": "Tensión Arterial Sistólica", "value": 150, "unit": "mmHg"}
        ],
        "allergies": [
            {"text": "Mariscos", "criticality": "high"}
        ],
        "meds": [
            {"text": "Insulina Glargina", "dosage": "18 unidades subcutáneas por la noche"},
            {"text": "Ramipril 10mg", "dosage": "1 comprimido al día"}
        ]
    },
    {
        "dni": "X7654321A",
        "given": "Sofia", "family": "Rossi", "birth": "1992-03-14", "gender": "female",
        "vitals": [
            {"code": "8867-4", "text": "Frecuencia Cardíaca", "value": 75, "unit": "bpm"},
            {"code": "2708-6", "text": "Saturación de Oxígeno (SpO2)", "value": 99, "unit": "%"}
        ],
        "allergies": [
            {"text": "Ibuprofeno y Naproxeno", "criticality": "high"}
        ],
        "meds": [
            {"text": "Nolotil (Metamizol) 575mg", "dosage": "1 cápsula si hay dolor intenso"}
        ]
    },
    {
        "dni": "Z9999999R",
        "given": "Mohamed", "family": "Alami", "birth": "1983-08-25", "gender": "male",
        "vitals": [
            {"code": "8480-6", "text": "Tensión Arterial Sistólica", "value": 120, "unit": "mmHg"},
            {"code": "8462-4", "text": "Tensión Arterial Diastólica", "value": 80, "unit": "mmHg"}
        ],
        "allergies": [
            {"text": "Gramíneas", "criticality": "low"}
        ],
        "meds": [
            {"text": "Spray Nasal Fluticasona", "dosage": "2 pulverizaciones por orificio nasal al día"}
        ]
    },
    {
        "dni": "PAT-8832",
        "given": "Carmen", "family": "Benítez Torres", "birth": "1970-05-19", "gender": "female",
        "vitals": [
            {"code": "8480-6", "text": "Tensión Arterial Sistólica", "value": 130, "unit": "mmHg"},
            {"code": "8867-4", "text": "Frecuencia Cardíaca", "value": 80, "unit": "bpm"}
        ],
        "allergies": [
            {"text": "Sulfamidas", "criticality": "low"}
        ],
        "meds": [
            {"text": "Omeprazol 20mg", "dosage": "1 cápsula antes del desayuno"}
        ]
    },
    {
        "dni": "12345",
        "given": "David", "family": "Sánchez Cano", "birth": "1995-10-02", "gender": "male",
        "vitals": [
            {"code": "8867-4", "text": "Frecuencia Cardíaca", "value": 58, "unit": "bpm"},
            {"code": "2708-6", "text": "Saturación de Oxígeno (SpO2)", "value": 98, "unit": "%"}
        ],
        "allergies": [
            {"text": "Melocotón (Proteína LTP)", "criticality": "low"}
        ],
        "meds": [
            {"text": "Complejo Vitamínico B", "dosage": "1 comprimido diario"}
        ]
    }
]

def post_resource(endpoint: str, payload: dict[str, Any]) -> str:
    url = f"{FHIR_BASE_URL}/{endpoint}"
    headers = {
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json"
    }
    with httpx.Client(timeout=20.0) as client:
        response = client.post(url, json=payload, headers=headers)
        
        if response.status_code in (400, 412) and "Patient/" in response.text:
            try:
                existing_id = response.text.split("Patient/")[1].split('"')[0].split("<")[0]
                print(f"  └─ ℹ️ El paciente ya existía en HAPI con ID: {existing_id}")
                return existing_id
            except Exception:
                pass
                
        response.raise_for_status()
        return response.json().get("id")

def create_patient_record(data: dict[str, Any]):
    print(f"\n⏳ Procesando expediente para {data['given']} {data['family']} (DNI: {data['dni']})...")
    
    unique_tag = str(uuid.uuid4())[:8]
    
    # 1. Patient
    patient_payload = {
        "resourceType": "Patient",
        "identifier": [{
            "system": "urn:dni-tfm",
            "value": f"{data['dni']}-{unique_tag}"
        }],
        "name": [{"given": [data["given"]], "family": f"{data['family']}"}],
        "birthDate": data["birth"],
        "gender": data["gender"]
    }
    patient_id = post_resource("Patient", patient_payload)
    print(f"  └─ ✅ Paciente asignado en HAPI con FHIR ID: {patient_id}")

    # 2. Vitals
    vitals_count = 0
    for v in data["vitals"]:
        try:
            obs_payload = {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "text": v["text"],
                    "coding": [{"system": "http://loinc.org", "code": v["code"], "display": v["text"]}]
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "valueQuantity": {"value": v["value"], "unit": v["unit"]}
            }
            post_resource("Observation", obs_payload)
            vitals_count += 1
        except Exception as e:
            print(f"     ⚠️ Error al guardar constante {v['text']}: {e}")
    print(f"  └─ ✅ {vitals_count} Observaciones añadidas.")

    # 3. Allergies
    allergies_count = 0
    for a in data["allergies"]:
        try:
            alg_payload = {
                "resourceType": "AllergyIntolerance",
                "clinicalStatus": {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]
                },
                "verificationStatus": {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]
                },
                "criticality": a["criticality"],  # Debe ser solo 'low' o 'high'
                "code": {
                    "text": a["text"],
                    "coding": [{"system": "http://snomed.info/sct", "display": a["text"]}]
                },
                "patient": {"reference": f"Patient/{patient_id}"}
            }
            post_resource("AllergyIntolerance", alg_payload)
            allergies_count += 1
        except Exception as e:
            print(f"     ⚠️ Error al guardar alergia {a['text']}: {e}")
    print(f"  └─ ✅ {allergies_count} Alergias registradas.")

    # 4. Medication
    meds_count = 0
    for m in data["meds"]:
        try:
            med_payload = {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "text": m["text"],
                    "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "display": m["text"]}]
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "dosageInstruction": [{"text": m["dosage"]}]
            }
            post_resource("MedicationRequest", med_payload)
            meds_count += 1
        except Exception as e:
            print(f"     ⚠️ Error al guardar medicación {m['text']}: {e}")
    print(f"  └─ ✅ {meds_count} Prescripciones registradas.")

    # 5. Documentos (DocumentReference)
    if "documents" in data:
        docs_count = 0
        for doc in data["documents"]:
            try:
                doc_payload = {
                    "resourceType": "DocumentReference",
                    "status": "current",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    # FHIR suele requerir el formato de fecha con hora/zona (ISO)
                    "date": f"{doc['date']}T08:00:00Z", 
                    "type": {
                        "text": doc["title"]
                    },
                    "content": [
                        {
                            "attachment": {
                                "contentType": doc["contentType"],
                                "url": doc["url"],
                                "title": doc["title"]
                            }
                        }
                    ]
                }
                post_resource("DocumentReference", doc_payload)
                docs_count += 1
            except Exception as e:
                print(f"     ⚠️ Error al guardar documento {doc['title']}: {e}")
        print(f"  └─ ✅ {docs_count} Documentos de pruebas registrados.")

    return patient_id

if __name__ == "__main__":
    print("🚀 Iniciando sembrado masivo de 10 pacientes en HAPI FHIR...")
    created_map = {}
    for p in PATIENTS_TO_CREATE:
        fhir_id = create_patient_record(p)
        created_map[p["dni"]] = fhir_id

    print("\n" + "="*50)
    print("🎉 ¡TODOS LOS PACIENTES HAN SIDO CREADOS CON ÉXITO!")
    print("="*50)
    print("\nCopia este diccionario de mapeo directamente a tu `app/mcp/server.py`:\n")
    print(f"PATIENT_REGISTRY = {created_map}")