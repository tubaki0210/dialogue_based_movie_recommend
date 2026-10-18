"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "system" | "user";
  content: string;
}

interface ChatViewProps {
  sessionId: string;
  initialMessage: string;
}

export default function ChatView({ sessionId, initialMessage }: ChatViewProps) {
  const [messages, setMessages] = useState<Message[]>([
    { role: "system", content: initialMessage },
  ]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !sessionId || loading) return;

    const userMsg = input.trim();
    setInput("");

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMsg },
      { role: "system", content: "" },
    ]);
    setLoading(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, user_input: userMsg }),
      });

      if (!response.body) throw new Error("レスポンスボディが空です。");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunkValue = decoder.decode(value, { stream: true });

        const lines = chunkValue.split("\n\n");
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith("data:")) {
            const dataStr = trimmedLine.replace("data:", "").trim();
            if (dataStr === "[DONE]") {
              done = true;
              break;
            }

            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.content) {
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  const lastIdx = newMsgs.length - 1;
                  newMsgs[lastIdx] = {
                    ...newMsgs[lastIdx],
                    content: newMsgs[lastIdx].content + parsed.content,
                  };
                  return newMsgs;
                });
              }
            } catch (jsonErr) {}
          }
        }
      }
    } catch (error) {
      console.error(error);
      alert("通信エラーが発生しました。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden border rounded-xl bg-gray-50 shadow-sm">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] p-3.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap shadow-sm ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-none"
                  : "bg-white border border-gray-200 text-gray-800 rounded-bl-none"
              }`}
            >
              {msg.content || (loading && idx === messages.length - 1 ? (
                <span className="animate-pulse text-gray-400">考え中...</span>
              ) : (
                ""
              ))}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="p-3 bg-white border-t flex gap-2 rounded-b-xl">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="好きなジャンルや感想を入力..."
          disabled={loading}
          className="flex-1 border rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-blue-600 text-white px-5 py.5 rounded-lg text-sm font-semibold hover:bg-blue-700 transition disabled:bg-gray-300"
        >
          送信
        </button>
      </form>
    </div>
  );
}