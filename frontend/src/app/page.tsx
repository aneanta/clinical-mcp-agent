"use client";

import { useState } from "react";
import { sendChatMessage } from "@/lib/api";

export interface MedicalDocument {
  id: string;
  title: string;
  content_type: string;
  url: string;
  date: string;
  category?: string;
}

interface Message {
  sender: "user" | "agent";
  text: string;
  documents?: MedicalDocument[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "agent",
      text: "Hola, doctor/a. Soy TFM Clinical Assistant. ¿Qué paciente o datos FHIR deseas consultar hoy?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Lista de todos los documentos del paciente activo y el documento seleccionado actualmente
  const [activeDocuments, setActiveDocuments] = useState<MedicalDocument[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<MedicalDocument | null>(null);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    setSelectedDoc(null);
    setActiveDocuments([]);

    const userText = input;
    setInput("");
    setMessages((prev) => [...prev, { sender: "user", text: userText }]);
    setLoading(true);

    try {
      const botResponse = await sendChatMessage(userText);

      let responseText = "";
      let responseDocs: MedicalDocument[] = [];

      // 1. Manejo seguro por si la respuesta viene como objeto o como string
      let data = botResponse;
      if (typeof botResponse === "string") {
        try {
          data = JSON.parse(botResponse);
        } catch {
          responseText = botResponse;
        }
      }

      // 2. Extraemos priorizando 'data.response' (que es la clave real que envía FastAPI)
      if (typeof data === "object" && data !== null) {
        responseText = data.response || data.text || data.message || "";
        responseDocs = data.documents || [];
      }

      // 3. Si la respuesta trae documentos, actualizamos la lista global
      if (responseDocs.length > 0) {
        const sortedDocs = [...responseDocs].sort(
          (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
        );
        setActiveDocuments(sortedDocs);
        setSelectedDoc(sortedDocs[0]);
      }

      setMessages((prev) => [
        ...prev,
        { sender: "agent", text: responseText, documents: responseDocs },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "agent", text: "⚠️ Error al conectar con el servidor MCP/FHIR." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex h-screen w-full bg-slate-900 text-slate-100 overflow-hidden">
      
      {/* ------------------------------------------------------------- */}
      {/* COLUMNA IZQUIERDA: CHATBOT                                   */}
      {/* ------------------------------------------------------------- */}
      <div
        className={`flex flex-col h-full p-6 transition-all duration-300 ${
          selectedDoc
            ? "w-full md:w-1/2 lg:w-5/12 border-r border-slate-800"
            : "w-full max-w-4xl mx-auto"
        }`}
      >
        <header className="border-b border-slate-700 pb-4 mb-4 flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold text-teal-400">TFM Clinical Assistant</h1>
            <p className="text-xs text-slate-400">Modo Local Dev • Connected via FastMCP & FHIR</p>
          </div>
          {selectedDoc && (
            <span className="text-xs bg-teal-950 text-teal-300 border border-teal-700 font-medium px-2.5 py-1 rounded-full">
              {activeDocuments.length} Documento(s)
            </span>
          )}
        </header>

        {/* Lista de Mensajes del Chat */}
        <div className="flex-1 overflow-y-auto space-y-4 pr-2">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-4 py-3 text-sm whitespace-pre-wrap ${
                  msg.sender === "user"
                    ? "bg-teal-600 text-white"
                    : "bg-slate-800 border border-slate-700 text-slate-200"
                }`}
              >
                {/* MODIFICACIÓN AQUÍ: Renderizado limpio de texto */}
                <div className="text-sm font-normal leading-relaxed">{msg.text}</div>

                {/* Muestra adjuntos ÚNICAMENTE si este mensaje concreto los contiene */}
                {Array.isArray(msg.documents) && msg.documents.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
                    <p className="text-xs text-teal-300 font-semibold">
                      📁 Pruebas de imagen / informes del paciente:
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {msg.documents.map((doc, idx) => {
                        // 1. Detección segura del tipo de contenido (imagen vs documento)
                        const contentType = doc?.content_type || doc?.mime_type || "";
                        const isImage = contentType.startsWith("image/");

                        // 2. Formateo seguro de la fecha
                        const formattedDate = doc?.date 
                          ? new Date(doc.date).toLocaleDateString() 
                          : "Sin fecha";

                        return (
                          <button
                            key={doc.id || `doc-${idx}`}
                            onClick={() => {
                              setActiveDocuments(msg.documents || []);
                              setSelectedDoc(doc);
                            }}
                            className={`text-xs p-2 rounded flex items-center justify-between border transition-all ${
                              selectedDoc?.id === doc.id
                                ? "bg-teal-950 border-teal-500 text-teal-200 font-semibold"
                                : "bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-500"
                            }`}
                          >
                            <div className="flex items-center gap-2 truncate">
                              <span>{isImage ? "🧠" : "📄"}</span>
                              <span className="truncate">{doc.title || doc.name || "Documento sin título"}</span>
                            </div>
                            <span className="text-[10px] text-slate-400 ml-2 whitespace-nowrap">
                              {formattedDate}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-800 border border-slate-700 text-slate-400 rounded-lg px-4 py-3 text-sm animate-pulse">
                Consultando expediente FHIR y repositorio de imágenes...
              </div>
            </div>
          )}
        </div>

        {/* Input del chat */}
        <form onSubmit={handleSend} className="mt-4 flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ej: 'Consultar paciente 137206566' o 'Muéstrame sus resonancias'..."
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors"
          >
            Enviar
          </button>
        </form>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* COLUMNA DERECHA: VISOR CLÍNICO LATERAL / HISTORIAL DE PRUEBAS   */}
      {/* ------------------------------------------------------------- */}
      {selectedDoc && (
        <div className="hidden md:flex md:w-1/2 lg:w-7/12 h-full flex-col bg-slate-950 border-l border-slate-800 text-slate-100 shadow-2xl">
          
          {/* Header del Visor */}
          <div className="p-4 bg-slate-900 border-b border-slate-800 flex justify-between items-center">
            <div>
              <h2 className="text-sm font-bold text-teal-400 truncate max-w-[350px]">
                {selectedDoc.title}
              </h2>
              <p className="text-xs text-slate-400">
                Fecha de la prueba: <span className="text-teal-300 font-medium">{selectedDoc.date}</span>
              </p>
            </div>

            <button
              onClick={() => {
                setSelectedDoc(null);
                setActiveDocuments([]);
              }}
              className="text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 p-1.5 rounded-lg text-xs transition"
              title="Cerrar visor"
            >
              ✕ Cerrar
            </button>
          </div>

          {/* Selector/Selector de Histórico si el paciente tiene más de 1 estudio */}
          {activeDocuments.length > 1 && (
            <div className="bg-slate-900/80 px-4 py-2 border-b border-slate-800 flex gap-2 overflow-x-auto items-center">
              <span className="text-xs text-slate-400 font-medium whitespace-nowrap mr-1">
                Histórico:
              </span>
              {activeDocuments.map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => setSelectedDoc(doc)}
                  className={`text-xs px-3 py-1 rounded-full whitespace-nowrap transition-colors flex items-center gap-1.5 ${
                    selectedDoc.id === doc.id
                      ? "bg-teal-600 text-white font-medium"
                      : "bg-slate-800 hover:bg-slate-700 text-slate-300"
                  }`}
                >
                  <span>{doc.content_type.startsWith("image/") ? "🖼️" : "📄"}</span>
                  <span>{new Date(doc.date).toLocaleDateString()}</span>
                </button>
              ))}
            </div>
          )}

          {/* Área de Visualización del Documento */}
          <div className="flex-1 p-4 overflow-auto flex items-center justify-center bg-slate-950 relative">
            {selectedDoc.content_type.startsWith("image/") ? (
              <div className="flex flex-col items-center justify-center h-full w-full">
                <img
                  src={selectedDoc.url}
                  alt={selectedDoc.title}
                  className="max-h-[78vh] max-w-full object-contain rounded border border-slate-800 shadow-2xl"
                />
              </div>
            ) : selectedDoc.content_type === "application/pdf" ? (
              <iframe
                src={selectedDoc.url}
                title={selectedDoc.title}
                className="w-full h-full min-h-[500px] rounded border-0 bg-white"
              />
            ) : (
              <div className="text-center p-6 bg-slate-900 rounded-lg border border-slate-800">
                <p className="text-sm text-slate-300 mb-4">
                  Vista previa no disponible para este formato.
                </p>
                <a
                  href={selectedDoc.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-teal-600 hover:bg-teal-500 text-white text-xs px-4 py-2 rounded font-medium shadow"
                >
                  Abrir enlace externo
                </a>
              </div>
            )}
          </div>
        </div>
      )}

    </main>
  );
}