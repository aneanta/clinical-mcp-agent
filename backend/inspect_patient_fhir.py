import json
from pathlib import Path


FHIR_FILE = Path(
    "../synthea/output/fhir/"
    "Miguel815_Bashirian201_5393ea7f-1fea-4064-9df4-460a2f662d07.json"
)


def get_display(codeable_concept):
    if not isinstance(codeable_concept, dict):
        return None

    if codeable_concept.get("text"):
        return codeable_concept["text"]

    for coding in codeable_concept.get("coding", []):
        if coding.get("display"):
            return coding["display"]

    return None


def get_observation_date(observation):
    return (
        observation.get("effectiveDateTime")
        or observation.get("issued")
        or observation.get("effectivePeriod", {}).get("start")
        or ""
    )


with open(
    FHIR_FILE,
    "r",
    encoding="utf-8"
) as file:
    bundle = json.load(file)


resources = [
    entry.get("resource", {})
    for entry in bundle.get("entry", [])
]


# ============================================================
# PATIENT
# ============================================================

patient = next(
    resource
    for resource in resources
    if resource.get("resourceType") == "Patient"
)

print("=" * 80)
print("PACIENTE")
print("=" * 80)

print("ID:", patient.get("id"))
print("Nacimiento:", patient.get("birthDate"))


# ============================================================
# MEDICATION REQUESTS
# ============================================================

medication_requests = [
    resource
    for resource in resources
    if resource.get("resourceType") == "MedicationRequest"
]

print()
print("=" * 80)
print("MEDICATION REQUESTS")
print("=" * 80)

print(f"Total: {len(medication_requests)}")


for medication_request in medication_requests:

    medication_name = get_display(
        medication_request.get(
            "medicationCodeableConcept"
        )
    )

    # Si el medicamento viene mediante referencia
    medication_reference = (
        medication_request
        .get("medicationReference", {})
        .get("reference")
    )

    print()
    print(
        "Status:",
        medication_request.get("status")
    )

    print(
        "Intent:",
        medication_request.get("intent")
    )

    print(
        "Medicamento:",
        medication_name
    )

    print(
        "Referencia:",
        medication_reference
    )

    print(
        "Fecha:",
        medication_request.get("authoredOn")
    )


# ============================================================
# LABORATORY OBSERVATIONS
# ============================================================

laboratory_observations = []

for resource in resources:

    if resource.get("resourceType") != "Observation":
        continue

    categories = resource.get("category", [])

    is_laboratory = any(
        coding.get("code") == "laboratory"
        for category in categories
        for coding in category.get("coding", [])
    )

    if is_laboratory:
        laboratory_observations.append(resource)


laboratory_observations.sort(
    key=get_observation_date,
    reverse=True
)


print()
print("=" * 80)
print("LABORATORY OBSERVATIONS")
print("=" * 80)

print(
    f"Total laboratorios: "
    f"{len(laboratory_observations)}"
)


# Mostramos los 30 más recientes
for observation in laboratory_observations[:30]:

    name = get_display(
        observation.get("code")
    )

    value_quantity = observation.get(
        "valueQuantity"
    )

    value_string = observation.get(
        "valueString"
    )

    print()
    print(
        "Prueba:",
        name
    )

    if value_quantity:
        print(
            "Valor:",
            value_quantity.get("value")
        )

        print(
            "Unidad:",
            value_quantity.get("unit")
        )

    elif value_string:
        print(
            "Valor:",
            value_string
        )

    else:
        print(
            "Valor:",
            "Formato no manejado todavía"
        )

    print(
        "Fecha:",
        get_observation_date(observation)
    )

    print(
        "Status:",
        observation.get("status")
    )