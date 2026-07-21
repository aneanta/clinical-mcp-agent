export async function sendChatMessage(message: string): Promise<string> {
  try {
    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`Error en el servidor: ${response.statusText}`);
    }

    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error("Error al enviar mensaje:", error);
    return "Error de conexión con el backend clínico. Asegúrate de que FastAPI esté corriendo en http://localhost:8000.";
  }
}
