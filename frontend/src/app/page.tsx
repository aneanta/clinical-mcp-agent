"use client";

import { useState } from "react";
import { sendChatMessage } from "@/lib/api";

interface Message {
  sender: "user" | "agent";
  text: string;
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

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input;
    setInput("");
    setMessages((prev) => [...prev, { sender: "user", text: userText }]);
    setLoading(true);

    const botResponse = await sendChatMessage(userText);
    setMessages((prev) => [...prev, { sender: "agent", text: botResponse }]);
    setLoading(false);
  };

  return (
    <main className="flex flex-col h-screen bg-slate-900 text-slate-100 p-6 max-w-4xl mx-auto">
      <header className="border-b border-slate-700 pb-4 mb-4">
        <h1 className="text-xl font-bold text-teal-400">TFM Clinical Assistant</h1>
        <p className="text-xs text-slate-400">Modo Local Dev • Connected via FastMCP & FHIR</p>
      </header>

      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-3 text-sm whitespace-pre-wrap ${
                msg.sender === "user"
                  ? "bg-teal-600 text-white"
                  : "bg-slate-800 border border-slate-700 text-slate-200"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 border border-slate-700 text-slate-400 rounded-lg px-4 py-3 text-sm animate-pulse">
              Consultando agente clínico y servidores MCP...
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSend} className="mt-4 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Escribe 'paciente' para ver ficha FHIR o tu consulta clínica..."
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
    </main>
  );
}
