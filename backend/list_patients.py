from pathlib import Path
import json
from datetime import date


FHIR_DIR = (
    Path(__file__).resolve().parent
    / ".."
    / "synthea"
    / "output"
    / "fhir"
).resolve()


def calculate_age(birth_date: str):
    if not birth_date:
        return None

    birth = date.fromisoformat(birth_date)
    today = date.today()

    return (
        today.year
        - birth.year
        - (
            (today.month, today.day)
            < (birth.month, birth.day)
        )
    )


def get_name(patient):
    names = patient.get("name", [])

    if not names:
        return "Sin nombre"

    name = names[0]

    given = " ".join(
        name.get("given", [])
    )

    family = name.get(
        "family",
        ""
    )

    return f"{given} {family}".strip()


patients = []

for file_path in FHIR_DIR.glob("*.json"):

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception:
        continue

    for entry in data.get("entry", []):

        resource = entry.get(
            "resource",
            {}
        )

        if resource.get("resourceType") != "Patient":
            continue

        patient_id = resource.get("id")

        name = get_name(resource)

        age = calculate_age(
            resource.get("birthDate")
        )

        patients.append(
            {
                "id": patient_id,
                "name": name,
                "age": age
            }
        )


for patient in patients[:20]:

    print(
        f"{patient['name']} "
        f"({patient['age']} años)"
    )

    print(
        f"ID: {patient['id']}"
    )

    print("-" * 70)