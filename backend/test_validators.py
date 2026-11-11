import unittest

from unittest.mock import patch, Mock

from multiagent_mcp import (
    validate_lab_interpretation,
    validate_fact_sources,
    validate_lab_units,
    clinical_validator,
    clinical_semantic_validator,
    after_semantic_validation,
    after_clinical_validation,
)

PATIENT_CONTEXT = {
    "get_lab_results": {
        "lab_results": {
            "glucose": 132,
            "cholesterol": 195
        }
    }
}


def build_analysis(hypothesis, based_on=None):
    if based_on is None:
        based_on = ["get_lab_results"]

    return {
        "known_facts": [],
        "clinical_reasoning": [
            {
                "hypothesis": hypothesis,
                "based_on": based_on
            }
        ],
        "additional_information": [],
        "response": ""
    }


class TestValidateLabInterpretation(unittest.TestCase):

    # =========================================================
    # CASOS QUE DEBEN SER RECHAZADOS
    # =========================================================

    def test_rejects_direct_hyperglycemia_interpretation(self):
        analysis = build_analysis(
            "La glucosa 132 indica hiperglucemia."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_elevated_even_with_warning_afterwards(self):
        analysis = build_analysis(
            "La glucosa 132 parece elevada, aunque faltan "
            "las unidades y el rango de referencia."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_glucose_alteration_as_fact(self):
        analysis = build_analysis(
            "Existe una alteración glucémica relacionada "
            "con el valor de glucosa."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertGreater(len(issues), 0)


    # =========================================================
    # CASOS QUE DEBEN SER ACEPTADOS
    # =========================================================

    def test_accepts_cannot_determine_hyperglycemia(self):
        analysis = build_analysis(
            "Sin unidad y rango de referencia no se puede "
            "determinar si el valor indica hiperglucemia o no."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertTrue(
            any(
                "utiliza get_lab_results como apoyo clínico"
                in issue
                for issue in issues
            )
        )


    def test_accepts_cannot_interpret_glucose(self):
        analysis = build_analysis(
            "El valor de glucosa 132 no puede interpretarse "
            "sin unidad y rango de referencia."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertTrue(
            any(
                "utiliza get_lab_results como apoyo clínico"
                in issue
                for issue in issues
            )
        )


    def test_accepts_cannot_conclude_elevated_or_normal(self):
        analysis = build_analysis(
            "El valor de glucosa de 132, sin contexto de ayuno "
            "o referencia, no permite concluir si es elevado "
            "o normal."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertTrue(
            any(
                "utiliza get_lab_results como apoyo clínico"
                in issue
                for issue in issues
            )
        )


    def test_accepts_plain_lab_value_without_interpretation(self):
        analysis = build_analysis(
            "El resultado disponible de glucosa es 132."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertTrue(
            any(
                "utiliza get_lab_results como apoyo clínico"
                in issue
                for issue in issues
            )
        )

    def test_does_not_duplicate_same_issue(self):
        analysis = build_analysis(
            "Existe una alteración glucémica relacionada "
            "con los resultados del laboratorio."
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertEqual(len(issues), 1)

    def test_accepts_interpretation_when_unit_and_range_exist(self):
        patient_context = {
            "get_lab_results": {
                "lab_results": {
                    "glucose": {
                        "value": 132,
                        "unit": "mg/dL",
                        "reference_range": "70-99"
                    }
                }
            }
        }

        analysis = build_analysis(
            "La glucosa está elevada."
        )

        issues = validate_lab_interpretation(
            analysis,
            patient_context
        )

        self.assertEqual(issues, [])

    def test_accepts_cannot_conclude_hyperglycemia_causes_symptoms(self):
        analysis = build_analysis(
            (
                "El valor de glucosa 132 no puede interpretarse sin unidad "
                "y rango de referencia, por lo tanto no se puede concluir "
                "que la sed y fatiga sean causadas por hiperglucemia."
            )
        )

        issues = validate_lab_interpretation(
            analysis,
            PATIENT_CONTEXT
        )

        self.assertTrue(
            any(
                "utiliza get_lab_results como apoyo clínico"
                in issue
                for issue in issues
            )
        )


class TestValidateFactSources(unittest.TestCase):

    PATIENT_CONTEXT = {
        "get_patient": {
            "name": "Laura",
            "age": 47
        },
        "get_medications": {
            "medications": ["Metformina", "Enalapril"]
        },
        "get_lab_results": {
            "lab_results": {
                "glucose": 132,
                "cholesterol": 195
            }
        }
    }

    QUESTION = (
        "El paciente 482 presenta mucha sed y fatiga."
    )

    def build_analysis(self, fact, source):
        return {
            "known_facts": [
                {
                    "fact": fact,
                    "source": source
                }
            ]
        }

    # =====================================================
    # CASOS CORRECTOS
    # =====================================================

    def test_accepts_correct_age(self):
        analysis = self.build_analysis(
            "Edad: 47 años",
            "get_patient"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertEqual(issues, [])


    def test_accepts_correct_name(self):
        analysis = self.build_analysis(
            "Nombre: Laura",
            "get_patient"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertEqual(issues, [])


    def test_accepts_correct_medication(self):
        analysis = self.build_analysis(
            "Medicamento actual: Metformina",
            "get_medications"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertEqual(issues, [])


    # =====================================================
    # CASOS QUE DEBEN RECHAZARSE
    # =====================================================

    def test_rejects_wrong_age(self):
        analysis = self.build_analysis(
            "Edad: 52 años",
            "get_patient"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_wrong_name(self):
        analysis = self.build_analysis(
            "Nombre: Marta",
            "get_patient"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_wrong_medication(self):
        analysis = self.build_analysis(
            "Medicamento actual: Ibuprofeno",
            "get_medications"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_fact_with_wrong_source(self):
        analysis = self.build_analysis(
            "Medicamento actual: Metformina",
            "get_patient"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_invented_lab_value(self):
        analysis = self.build_analysis(
            "Glucosa: 999",
            "get_lab_results"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_diagnosis_as_medication_fact(self):
        analysis = self.build_analysis(
            "La paciente tiene diabetes",
            "get_medications"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


    def test_accepts_correct_lab_value(self):
        analysis = self.build_analysis(
            "Glucosa: 132",
            "get_lab_results"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertEqual(issues, [])


    def test_accepts_fact_from_user_question(self):
        analysis = self.build_analysis(
            "El paciente presenta mucha sed y fatiga",
            "user_question"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertEqual(issues, [])


    def test_rejects_invented_user_question_fact(self):
        analysis = self.build_analysis(
            "El paciente presenta fiebre",
            "user_question"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_unknown_source(self):
        analysis = self.build_analysis(
            "Edad: 47 años",
            "medical_record"
        )

        issues = validate_fact_sources(
            analysis,
            self.PATIENT_CONTEXT,
            self.QUESTION
        )

        self.assertGreater(len(issues), 0)


class TestClinicalValidator(unittest.TestCase):

    def run_validator(self, analysis):
        state = {
            "clinical_analysis": analysis,
            "clinical_structure_retry_count": 0
        }

        return clinical_validator(state)

    def valid_analysis(self):
        return {
            "known_facts": [
                {
                    "fact": "Edad: 47 años",
                    "source": "get_patient"
                }
            ],
            "clinical_reasoning": [
                {
                    "hypothesis": "La sed podría estar relacionada con deshidratación.",
                    "based_on": ["user_question"]
                }
            ],
            "additional_information": [
                "Frecuencia urinaria"
            ],
            "response": (
                "Sería útil conocer la frecuencia urinaria."
            )
        }

    # =====================================================
    # ESTRUCTURA GENERAL
    # =====================================================

    def test_accepts_valid_clinical_analysis(self):
        result = self.run_validator(
            self.valid_analysis()
        )

        self.assertTrue(
            result["clinical_validation_ok"]
        )

        self.assertEqual(
            result["clinical_validation_issues"],
            []
        )


    def test_rejects_non_dict_analysis(self):
        result = self.run_validator(
            "respuesta incorrecta"
        )

        self.assertFalse(
            result["clinical_validation_ok"]
        )


    def test_rejects_missing_response(self):
        analysis = self.valid_analysis()
        del analysis["response"]

        result = self.run_validator(analysis)

        self.assertFalse(
            result["clinical_validation_ok"]
        )


    # =====================================================
    # ESTRUCTURA INTERNA
    # =====================================================

    def test_rejects_known_fact_without_structure(self):
        analysis = self.valid_analysis()

        analysis["known_facts"] = [
            "Edad: 47 años"
        ]

        result = self.run_validator(analysis)

        self.assertFalse(
            result["clinical_validation_ok"]
        )


    def test_rejects_known_fact_without_source(self):
        analysis = self.valid_analysis()

        analysis["known_facts"] = [
            {
                "fact": "Edad: 47 años"
            }
        ]

        result = self.run_validator(analysis)

        self.assertFalse(
            result["clinical_validation_ok"]
        )


    def test_rejects_reasoning_without_based_on(self):
        analysis = self.valid_analysis()

        analysis["clinical_reasoning"] = [
            {
                "hypothesis": (
                    "La sed podría estar relacionada "
                    "con deshidratación."
                )
            }
        ]

        result = self.run_validator(analysis)

        self.assertFalse(
            result["clinical_validation_ok"]
        )


    def test_rejects_reasoning_as_plain_string(self):
        analysis = self.valid_analysis()

        analysis["clinical_reasoning"] = [
            "Posible deshidratación"
        ]

        result = self.run_validator(analysis)

        self.assertFalse(
            result["clinical_validation_ok"]
        )


class TestValidateLabUnits(unittest.TestCase):

    def analysis_with_fact(self, fact):
        return {
            "known_facts": [
                {
                    "fact": fact,
                    "source": "get_lab_results"
                }
            ],
            "clinical_reasoning": [],
            "additional_information": [],
            "response": ""
        }

    def test_accepts_value_without_unit(self):
        patient_context = {
            "get_lab_results": {
                "lab_results": {
                    "glucose": 132
                }
            }
        }

        analysis = self.analysis_with_fact(
            "Glucosa: 132"
        )

        issues = validate_lab_units(
            analysis,
            patient_context
        )

        self.assertEqual(issues, [])


    def test_rejects_invented_mg_dl_unit(self):
        patient_context = {
            "get_lab_results": {
                "lab_results": {
                    "glucose": 132
                }
            }
        }

        analysis = self.analysis_with_fact(
            "Glucosa: 132 mg/dL"
        )

        issues = validate_lab_units(
            analysis,
            patient_context
        )

        self.assertGreater(len(issues), 0)


    def test_rejects_invented_unit_in_response(self):
        patient_context = {
            "get_lab_results": {
                "lab_results": {
                    "glucose": 132
                }
            }
        }

        analysis = {
            "known_facts": [],
            "clinical_reasoning": [],
            "additional_information": [],
            "response": (
                "La glucosa registrada es de 132 mmol/L."
            )
        }

        issues = validate_lab_units(
            analysis,
            patient_context
        )

        self.assertGreater(len(issues), 0)


    def test_accepts_unit_when_provided_by_source(self):
        patient_context = {
            "get_lab_results": {
                "lab_results": {
                    "glucose": {
                        "value": 132,
                        "unit": "mg/dL",
                        "reference_range": "70-99"
                    }
                }
            }
        }

        analysis = self.analysis_with_fact(
            "Glucosa: 132 mg/dL"
        )

        issues = validate_lab_units(
            analysis,
            patient_context
        )

        self.assertEqual(issues, [])


    def test_rejects_invented_unit_in_structured_lab_without_unit(self):
        patient_context = {
            "get_lab_results": {
                "lab_results": {
                    "glucose": {
                        "value": 132,
                        "reference_range": "70-99"
                    }
                }
            }
        }

        analysis = self.analysis_with_fact(
            "Glucosa: 132 mg/dL"
        )

        issues = validate_lab_units(
            analysis,
            patient_context
        )

        self.assertGreater(len(issues), 0)

    def test_rejects_different_unit_than_source(self):
        patient_context = {
            "get_lab_results": {
                "lab_results": {
                    "glucose": {
                        "value": 132,
                        "unit": "mg/dL",
                        "reference_range": "70-99"
                    }
                }
            }
        }

        analysis = self.analysis_with_fact(
            "Glucosa: 132 mmol/L"
        )

        issues = validate_lab_units(
            analysis,
            patient_context
        )

        self.assertGreater(len(issues), 0)
        

class TestClinicalSemanticValidator(unittest.TestCase):

    def valid_state(self):
        return {
            "question": (
                "El paciente 482 presenta mucha sed y fatiga."
            ),
            "patient_context": {
                "get_patient": {
                    "name": "Laura",
                    "age": 47
                },
                "get_medications": {
                    "medications": [
                        "Metformina",
                        "Enalapril"
                    ]
                },
                "get_lab_results": {
                    "lab_results": {
                        "glucose": 132,
                        "cholesterol": 195
                    }
                }
            },
            "symptom_analysis": {
                "symptoms": [
                    "sed",
                    "fatiga"
                ],
                "questions": [
                    "¿Ha aumentado la frecuencia urinaria?"
                ]
            },
            "clinical_analysis": {
                "known_facts": [
                    {
                        "fact": "Edad: 47 años",
                        "source": "get_patient"
                    },
                    {
                        "fact": "Nombre: Laura",
                        "source": "get_patient"
                    },
                    {
                        "fact": "Medicamento actual: Metformina",
                        "source": "get_medications"
                    },
                    {
                        "fact": "Glucosa: 132",
                        "source": "get_lab_results"
                    },
                    {
                        "fact": (
                            "El paciente presenta mucha sed y fatiga"
                        ),
                        "source": "user_question"
                    }
                ],
                "clinical_reasoning": [
                    {
                        "hypothesis": (
                            "La sed podría estar relacionada "
                            "con deshidratación."
                        ),
                        "based_on": [
                            "user_question"
                        ]
                    }
                ],
                "additional_information": [
                    "Frecuencia urinaria"
                ],
                "response": (
                    "Sería útil conocer la frecuencia urinaria."
                )
            },
            "clinical_retry_count": 0
        }


    def mock_response(self, content):
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = content

        return response


    # =====================================================
    # TODO CORRECTO
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_accepts_when_llm_and_deterministic_validators_pass(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            '{"valid": true, "issues": []}'
        )

        state = self.valid_state()

        result = clinical_semantic_validator(state)

        self.assertTrue(
            result["semantic_validation_ok"]
        )

        self.assertEqual(
            result["semantic_validation_issues"],
            []
        )

        self.assertEqual(
            result["clinical_retry_count"],
            0
        )


    # =====================================================
    # GROQ DICE OK, PERO HAY UNIDAD INVENTADA
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_rejects_invented_unit_even_if_llm_says_valid(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            '{"valid": true, "issues": []}'
        )

        state = self.valid_state()

        state["clinical_analysis"]["known_facts"][3][
            "fact"
        ] = "Glucosa: 132 mg/dL"

        result = clinical_semantic_validator(state)

        self.assertFalse(
            result["semantic_validation_ok"]
        )

        self.assertGreater(
            len(result["semantic_validation_issues"]),
            0
        )

        self.assertEqual(
            result["clinical_retry_count"],
            1
        )


    # =====================================================
    # GROQ DICE OK, PERO HAY UN HECHO INVENTADO
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_rejects_invented_fact_even_if_llm_says_valid(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            '{"valid": true, "issues": []}'
        )

        state = self.valid_state()

        state["clinical_analysis"]["known_facts"][0][
            "fact"
        ] = "Edad: 52 años"

        result = clinical_semantic_validator(state)

        self.assertFalse(
            result["semantic_validation_ok"]
        )

        self.assertGreater(
            len(result["semantic_validation_issues"]),
            0
        )

        self.assertEqual(
            result["clinical_retry_count"],
            1
        )


    # =====================================================
    # EL LLM DETECTA UN PROBLEMA
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_rejects_when_llm_validator_says_invalid(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            (
                '{"valid": false, '
                '"issues": ["Diagnóstico definitivo detectado."]}'
            )
        )

        state = self.valid_state()

        result = clinical_semantic_validator(state)

        self.assertFalse(
            result["semantic_validation_ok"]
        )

        self.assertIn(
            "Diagnóstico definitivo detectado.",
            result["semantic_validation_issues"]
        )

        self.assertEqual(
            result["clinical_retry_count"],
            1
        )


    # =====================================================
    # GROQ DEVUELVE JSON INVÁLIDO
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_rejects_invalid_json_from_llm_validator(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            "Esto no es JSON"
        )

        state = self.valid_state()

        result = clinical_semantic_validator(state)

        self.assertFalse(
            result["semantic_validation_ok"]
        )

        self.assertIn(
            (
                "El validador semántico no devolvió "
                "JSON válido."
            ),
            result["semantic_validation_issues"]
        )

        self.assertEqual(
            result["clinical_retry_count"],
            1
        )


    # =====================================================
    # COMBINA PROBLEMAS LLM + DETERMINISTAS
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_combines_llm_and_deterministic_issues(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            (
                '{"valid": false, '
                '"issues": ["Diagnóstico definitivo detectado."]}'
            )
        )

        state = self.valid_state()

        state["clinical_analysis"]["known_facts"][3][
            "fact"
        ] = "Glucosa: 132 mg/dL"

        result = clinical_semantic_validator(state)

        issues = result[
            "semantic_validation_issues"
        ]

        self.assertFalse(
            result["semantic_validation_ok"]
        )

        self.assertTrue(
            any(
                "Diagnóstico definitivo" in issue
                for issue in issues
            )
        )

        self.assertTrue(
            any(
                "unidad no proporcionada" in issue.lower()
                for issue in issues
            )
        )

        self.assertEqual(
            result["clinical_retry_count"],
            1
        )

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_rejects_diagnosis_inferred_from_medication(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            '{"valid": true, "issues": []}'
        )

        state = self.valid_state()

        state["clinical_analysis"]["clinical_reasoning"] = [
            {
                "hypothesis": (
                    "El uso de Metformina sugiere manejo "
                    "de diabetes."
                ),
                "based_on": [
                    "get_medications"
                ]
            }
        ]

        result = clinical_semantic_validator(state)

        self.assertFalse(
            result["semantic_validation_ok"]
        )

        self.assertGreater(
            len(result["semantic_validation_issues"]),
            0
        )


class TestClinicalRetryRouting(unittest.TestCase):

    # =====================================================
    # VALIDACIÓN SEMÁNTICA
    # =====================================================

    def test_semantic_valid_continues(self):
        state = {
            "semantic_validation_ok": True,
            "clinical_retry_count": 0
        }

        result = after_semantic_validation(state)

        self.assertEqual(result, "valid")


    def test_semantic_invalid_retries_before_limit(self):
        state = {
            "semantic_validation_ok": False,
            "clinical_retry_count": 1
        }

        result = after_semantic_validation(state)

        self.assertEqual(result, "retry")


    def test_semantic_invalid_fails_at_limit(self):
        state = {
            "semantic_validation_ok": False,
            "clinical_retry_count": 2
        }

        result = after_semantic_validation(state)

        self.assertEqual(result, "failed")


    # =====================================================
    # VALIDACIÓN ESTRUCTURAL
    # =====================================================

    def test_structure_valid_continues(self):
        state = {
            "clinical_validation_ok": True,
            "clinical_structure_retry_count": 0
        }

        result = after_clinical_validation(state)

        self.assertEqual(result, "continue")


    def test_structure_invalid_retries_before_limit(self):
        state = {
            "clinical_validation_ok": False,
            "clinical_structure_retry_count": 1
        }

        result = after_clinical_validation(state)

        self.assertEqual(result, "retry")


    def test_structure_invalid_fails_at_limit(self):
        state = {
            "clinical_validation_ok": False,
            "clinical_structure_retry_count": 2
        }

        result = after_clinical_validation(state)

        self.assertEqual(result, "failed")

    def test_semantic_retry_ignores_structure_retry_count(self):
        state = {
            "semantic_validation_ok": False,
            "clinical_retry_count": 0,
            "clinical_structure_retry_count": 99
        }

        result = after_semantic_validation(state)

        self.assertEqual(result, "retry")


    def test_semantic_failure_depends_only_on_semantic_counter(self):
        state = {
            "semantic_validation_ok": False,
            "clinical_retry_count": 2,
            "clinical_structure_retry_count": 0
        }

        result = after_semantic_validation(state)

        self.assertEqual(result, "failed")


    def test_structure_retry_ignores_semantic_retry_count(self):
        state = {
            "clinical_validation_ok": False,
            "clinical_structure_retry_count": 0,
            "clinical_retry_count": 99
        }

        result = after_clinical_validation(state)

        self.assertEqual(result, "retry")


    def test_structure_failure_depends_only_on_structure_counter(self):
        state = {
            "clinical_validation_ok": False,
            "clinical_structure_retry_count": 2,
            "clinical_retry_count": 0
        }

        result = after_clinical_validation(state)

        self.assertEqual(result, "failed")


class TestClinicalValidationFlow(unittest.TestCase):

    def valid_state(self):
        return {
            "question": (
                "El paciente 482 presenta mucha sed y fatiga."
            ),
            "patient_context": {
                "get_patient": {
                    "name": "Laura",
                    "age": 47
                },
                "get_medications": {
                    "medications": [
                        "Metformina",
                        "Enalapril"
                    ]
                },
                "get_lab_results": {
                    "lab_results": {
                        "glucose": 132,
                        "cholesterol": 195
                    }
                }
            },
            "symptom_analysis": {
                "symptoms": ["sed", "fatiga"],
                "questions": [
                    "¿Ha aumentado la frecuencia urinaria?"
                ]
            },
            "clinical_analysis": {
                "known_facts": [
                    {
                        "fact": "Edad: 47 años",
                        "source": "get_patient"
                    },
                    {
                        "fact": "Glucosa: 132",
                        "source": "get_lab_results"
                    }
                ],
                "clinical_reasoning": [
                    {
                        "hypothesis": (
                            "La sed podría estar relacionada "
                            "con deshidratación."
                        ),
                        "based_on": [
                            "user_question"
                        ]
                    }
                ],
                "additional_information": [
                    "Frecuencia urinaria"
                ],
                "response": (
                    "Sería útil conocer la frecuencia urinaria."
                )
            },
            "clinical_structure_retry_count": 0,
            "clinical_retry_count": 0
        }


    def mock_response(self, content):
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = content
        return response


    # =====================================================
    # FLUJO COMPLETO CORRECTO
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_valid_structure_then_valid_semantics(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            '{"valid": true, "issues": []}'
        )

        state = self.valid_state()

        # 1. Validador estructural
        structure_result = clinical_validator(state)

        state.update(structure_result)

        structure_route = after_clinical_validation(
            state
        )

        self.assertEqual(
            structure_route,
            "continue"
        )

        # 2. Validador semántico
        semantic_result = clinical_semantic_validator(
            state
        )

        state.update(semantic_result)

        semantic_route = after_semantic_validation(
            state
        )

        self.assertEqual(
            semantic_route,
            "valid"
        )

        self.assertEqual(
            state["clinical_structure_retry_count"],
            0
        )

        self.assertEqual(
            state["clinical_retry_count"],
            0
        )


    # =====================================================
    # FALLO ESTRUCTURAL
    # =====================================================

    def test_structure_failure_only_increments_structure_counter(
        self
    ):
        state = self.valid_state()

        # Rompemos la estructura
        state["clinical_analysis"]["known_facts"] = [
            "Edad: 47 años"
        ]

        result = clinical_validator(state)

        state.update(result)

        route = after_clinical_validation(state)

        self.assertEqual(
            route,
            "retry"
        )

        self.assertEqual(
            state["clinical_structure_retry_count"],
            1
        )

        self.assertEqual(
            state["clinical_retry_count"],
            0
        )


    # =====================================================
    # FALLO SEMÁNTICO
    # =====================================================

    @patch(
        "multiagent_mcp.groq_client.chat.completions.create"
    )
    def test_semantic_failure_only_increments_semantic_counter(
        self,
        mock_create
    ):
        mock_create.return_value = self.mock_response(
            '{"valid": true, "issues": []}'
        )

        state = self.valid_state()

        # Estructura válida
        structure_result = clinical_validator(state)

        state.update(structure_result)

        self.assertEqual(
            after_clinical_validation(state),
            "continue"
        )

        # Introducimos un error semántico después
        state["clinical_analysis"]["known_facts"][1][
            "fact"
        ] = "Glucosa: 132 mg/dL"

        semantic_result = clinical_semantic_validator(
            state
        )

        state.update(semantic_result)

        route = after_semantic_validation(state)

        self.assertEqual(
            route,
            "retry"
        )

        self.assertEqual(
            state["clinical_structure_retry_count"],
            0
        )

        self.assertEqual(
            state["clinical_retry_count"],
            1
        )


if __name__ == "__main__":
    unittest.main()