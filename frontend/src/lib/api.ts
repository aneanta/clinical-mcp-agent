// src/lib/api.ts
export async function sendChatMessage(message: string) {
  const response = await fetch("http://localhost:8000/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error("Error en la petición al servidor");
  }

  // ⚠️ IMPORTANTE: Retornar todo el objeto JSON, NO solo .response o .text
  const data = await response.json();
  return data; 
}