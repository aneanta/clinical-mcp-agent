// src/lib/api.ts
export interface ChatResponse {
  response: string;
  route: string | null;
  required_data: string[];
  patient_context: Record<string, unknown>;
  symptom_analysis: Record<string, unknown>;
  clinical_analysis: Record<string, unknown>;
  last_patient_id: string | null;
  documents?: unknown[];
}

export async function sendChatMessage(
  message: string,
  lastPatientId: string | null
): Promise<ChatResponse> {
  const response = await fetch("http://127.0.0.1:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, last_patient_id: lastPatientId }),
  });

  if (!response.ok) {
    throw new Error("Error en la petición al servidor");
  }

  return response.json();
}