import json
from pathlib import Path
from collections import Counter


FHIR_DIR = Path("../synthea/output/fhir")


def get_patient_name(patient):
    names = patient.get("name", [])

    if not names:
        return "Sin nombre"

    name = names[0]

    given = " ".join(name.get("given", []))
    family = name.get("family", "")

    return f"{given} {family}".strip()


for file_path in FHIR_DIR.glob("*.json"):

    # Ignoramos los ficheros generales de hospitales y profesionales
    if (
        file_path.name.startswith("hospitalInformation")
        or file_path.name.startswith("practitionerInformation")
    ):
        continue

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        bundle = json.load(file)

    resources = [
        entry.get("resource", {})
        for entry in bundle.get("entry", [])
    ]

    patient = next(
        (
            resource
            for resource in resources
            if resource.get("resourceType") == "Patient"
        ),
        None
    )

    if not patient:
        continue

    resource_types = Counter(
        resource.get("resourceType")
        for resource in resources
    )

    print("=" * 70)
    print(f"Archivo: {file_path.name}")
    print(f"FHIR Patient ID: {patient.get('id')}")
    print(f"Nombre: {get_patient_name(patient)}")
    print(f"Fecha nacimiento: {patient.get('birthDate')}")
    print()
    print("Recursos:")

    for resource_type, count in sorted(
        resource_types.items()
    ):
        print(
            f"  {resource_type}: {count}"
        )