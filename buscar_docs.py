import httpx

# Servidor público oficial de pruebas FHIR R4
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

def encontrar_pacientes_con_documentos():
    print("Buscando los últimos documentos en el servidor HAPI FHIR...")
    
    # Hacemos una petición directa a DocumentReference (sin filtrar por paciente)
    # Traemos los 50 más recientes
    url = f"{FHIR_BASE_URL}/DocumentReference"
    
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, headers={"Accept": "application/fhir+json"}, params={"_count": 50})
            response.raise_for_status()
            data = response.json()
            
            entries = data.get("entry", [])
            print(f"Se han analizado {len(entries)} documentos recientes.\n")
            
            pacientes_encontrados = set()
            
            for entry in entries:
                res = entry.get("resource", {})
                if res.get("resourceType") == "DocumentReference":
                    
                    # 1. Obtener a quién pertenece (subject / patient)
                    subject = res.get("subject", {})
                    patient_ref = subject.get("reference", "")
                    
                    # Limpiamos el ID (suele venir como "Patient/123456")
                    patient_id = patient_ref.replace("Patient/", "") if patient_ref else "Desconocido"
                    
                    # 2. Comprobar si tiene un attachment con URL
                    content = res.get("content", [])
                    has_url = False
                    file_type = "Desconocido"
                    url_adjunto = ""
                    
                    if content:
                        attachment = content[0].get("attachment", {})
                        url_adjunto = attachment.get("url", "")
                        file_type = attachment.get("contentType", "application/pdf")
                        if url_adjunto:
                            has_url = True
                    
                    # Si tiene un paciente asignado y una URL de archivo válida, lo mostramos
                    if patient_id != "Desconocido" and has_url:
                        if patient_id not in pacientes_encontrados:
                            print("-" * 50)
                            print(f"✅ ¡Paciente encontrado! ID en FHIR: {patient_id}")
                            pacientes_encontrados.add(patient_id)
                        
                        print(f"   📄 Tipo: {file_type}")
                        print(f"   🔗 URL: {url_adjunto}")
                        
            print("-" * 50)
            print(f"\nResumen: Se encontraron {len(pacientes_encontrados)} pacientes distintos con documentos accesibles.")
            print("Consejo: Copia uno de esos 'ID en FHIR' y añádelo a tu PATIENT_REGISTRY en server.py")
            
    except Exception as e:
        print(f"Error al conectar con FHIR: {e}")

if __name__ == "__main__":
    encontrar_pacientes_con_documentos()