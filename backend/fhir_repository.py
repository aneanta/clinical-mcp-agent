import json
import re
from datetime import date
from pathlib import Path


FHIR_DIR = (
    Path(__file__).resolve().parents[2]
    / "synthea"
    / "output"
    / "fhir"
)

TEST_REFERENCE_RANGES = {
    "Glucose [Mass/volume] in Blood": "70 - 99 mg/dL",
    "Creatinine [Mass/volume] in Blood": "0.6 - 1.3 mg/dL",
    "Sodium [Moles/volume] in Blood": "135 - 145 mmol/L",
    "Potassium [Moles/volume] in Blood": "3.5 - 5.0 mmol/L",
    "Calcium [Mass/volume] in Blood": "8.5 - 10.5 mg/dL",
}

# ============================================================
# HELPERS
# ============================================================

def _get_display(codeable_concept):
    if not isinstance(codeable_concept, dict):
        return None

    if codeable_concept.get("text"):
        return codeable_concept["text"]

    for coding in codeable_concept.get("coding", []):
        if coding.get("display"):
            return coding["display"]

    return None


def _calculate_age(birth_date):
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


def _get_patient_name(patient):
    names = patient.get("name", [])

    if not names:
        return None

    # Preferimos el nombre oficial si existe
    name = next(
        (
            item
            for item in names
            if item.get("use") == "official"
        ),
        names[0]
    )

    given = " ".join(name.get("given", []))
    family = name.get("family", "")

    return f"{given} {family}".strip()


def _get_observation_date(observation):
    return (
        observation.get("effectiveDateTime")
        or observation.get("issued")
        or observation.get(
            "effectivePeriod",
            {}
        ).get("start")
        or ""
    )


def _is_laboratory_observation(observation):
    for category in observation.get("category", []):
        for coding in category.get("coding", []):
            if coding.get("code") == "laboratory":
                return True

    return False


def _extract_reference_range(observation):
    ranges = observation.get("referenceRange", [])

    if not ranges:
        return None

    reference = ranges[0]

    low = reference.get("low", {})
    high = reference.get("high", {})

    low_value = low.get("value")
    high_value = high.get("value")

    unit = (
        low.get("unit")
        or high.get("unit")
    )

    if (
        low_value is not None
        and high_value is not None
    ):
        if unit:
            return (
                f"{low_value} - "
                f"{high_value} {unit}"
            )

        return f"{low_value} - {high_value}"

    if reference.get("text"):
        return reference["text"]

    return None


# ============================================================
# CARGA DEL BUNDLE
# ============================================================

def _load_patient_bundle(patient_id):

    for file_path in FHIR_DIR.glob("*.json"):

        if (
            file_path.name.startswith(
                "hospitalInformation"
            )
            or file_path.name.startswith(
                "practitionerInformation"
            )
        ):
            continue

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            bundle = json.load(file)

        for entry in bundle.get("entry", []):
            resource = entry.get(
                "resource",
                {}
            )

            if (
                resource.get("resourceType")
                == "Patient"
                and resource.get("id")
                == patient_id
            ):
                return bundle

    return None


def _get_resources(bundle):

    return [
        entry.get("resource", {})
        for entry in bundle.get("entry", [])
    ]


def _build_reference_index(bundle):
    """
    Permite resolver referencias como:
    urn:uuid:11d6acc7-...
    """

    index = {}

    for entry in bundle.get("entry", []):

        resource = entry.get(
            "resource",
            {}
        )

        full_url = entry.get("fullUrl")

        if full_url:
            index[full_url] = resource

        resource_id = resource.get("id")

        if resource_id:
            index[resource_id] = resource

            index[
                f"urn:uuid:{resource_id}"
            ] = resource

    return index


# ============================================================
# PATIENT
# ============================================================

def get_patient(patient_id):

    bundle = _load_patient_bundle(
        patient_id
    )

    if not bundle:
        return {
            "error": "Paciente no encontrado."
        }

    resources = _get_resources(bundle)

    patient = next(
        (
            resource
            for resource in resources
            if resource.get(
                "resourceType"
            ) == "Patient"
        ),
        None
    )

    if not patient:
        return {
            "error": (
                "El Bundle no contiene "
                "un recurso Patient."
            )
        }

    birth_date = patient.get(
        "birthDate"
    )

    return {
        "name": _get_patient_name(patient),
        "age": _calculate_age(birth_date),
        "birth_date": birth_date
    }


# ============================================================
# MEDICATIONS
# ============================================================

def get_medications(patient_id):

    bundle = _load_patient_bundle(
        patient_id
    )

    if not bundle:
        return {
            "error": "Paciente no encontrado."
        }

    resources = _get_resources(bundle)
    reference_index = (
        _build_reference_index(bundle)
    )

    medication_requests = [
        resource
        for resource in resources
        if (
            resource.get("resourceType")
            == "MedicationRequest"
            and resource.get("status")
            == "active"
        )
    ]

    medications = []

    for request in medication_requests:

        name = _get_display(
            request.get(
                "medicationCodeableConcept"
            )
        )

        # Si FHIR utiliza medicationReference
        if not name:

            reference = (
                request
                .get(
                    "medicationReference",
                    {}
                )
                .get("reference")
            )

            if reference:

                medication_resource = (
                    reference_index.get(
                        reference
                    )
                )

                if medication_resource:
                    name = _get_display(
                        medication_resource.get(
                            "code"
                        )
                    )

        if name:
            medications.append(name)

    # Quitamos duplicados conservando orden
    medications = list(
        dict.fromkeys(medications)
    )

    return {
        "medications": medications
    }


# ============================================================
# LAB RESULTS
# ============================================================

def get_lab_results(patient_id):

    bundle = _load_patient_bundle(
        patient_id
    )

    if not bundle:
        return {
            "error": "Paciente no encontrado."
        }

    resources = _get_resources(bundle)

    observations = [
        resource
        for resource in resources
        if (
            resource.get("resourceType")
            == "Observation"
            and _is_laboratory_observation(
                resource
            )
            and resource.get("status")
            in {
                "final",
                "amended",
                "corrected"
            }
        )
    ]

    # De momento trabajamos solo con
    # resultados numéricos.
    observations = [
        observation
        for observation in observations
        if isinstance(
            observation.get(
                "valueQuantity"
            ),
            dict
        )
    ]

    if not observations:
        return {
            "lab_results": {}
        }

    # Queremos el conjunto de resultados
    # correspondiente a la fecha más reciente.
    latest_date = max(
        _get_observation_date(
            observation
        )
        for observation
        in observations
    )

    latest_observations = [
        observation
        for observation in observations
        if _get_observation_date(
            observation
        ) == latest_date
    ]

    lab_results = {}

    for observation in latest_observations:

        name = _get_display(
            observation.get("code")
        )

        if not name:
            continue

        quantity = observation.get(
            "valueQuantity",
            {}
        )

        value = quantity.get("value")
        unit = quantity.get("unit")

        reference_range = _extract_reference_range(
            observation
        )

        if reference_range is not None:
            reference_range_source = "fhir"
        else:
            reference_range = TEST_REFERENCE_RANGES.get(
                name
            )

            if reference_range is not None:
                reference_range_source = "synthetic_test"
            else:
                reference_range_source = None

        lab_results[name] = {
            "value": value,
            "unit": unit,
            "reference_range": reference_range,
            "reference_range_source": reference_range_source,
            "date": (
                _get_observation_date(
                    observation
                )
            )
        }

    return {
        "lab_results": lab_results
    }