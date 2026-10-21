"use client";

import { useState } from "react";

interface LoginViewProps {
  onSessionStarted: (sessionId: string) => void;
}

export default function LoginView({ onSessionStarted }: LoginViewProps) {
  const [userId, setUserId] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const handleStartSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId.trim()) return;

    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId.trim() }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "セッションの開始に失敗しました。");
      }

      const data = await res.json();
      onSessionStarted(data.session_id); 
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col justify-center items-center flex-1">
      <form onSubmit={handleStartSession} className="w-full max-w-md flex flex-col gap-4 p-6 border rounded-xl shadow-md bg-gray-50">
        <label className="text-sm font-semibold text-gray-700">
          ユーザーIDを入力してください：
        </label>
        <input
          type="text"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          className="border p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="例: 147288"
          required
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white p-2.5 rounded-lg font-semibold hover:bg-blue-700 transition disabled:bg-gray-400"
        >
          {loading ? "セッション開始中..." : "チャットを開始"}
        </button>
      </form>
    </div>
  );
}