"use client";

import { useContext, useState } from "react";
import { UserIdContext, useUserId } from "./UserIdProvider";

interface LoginViewProps {
  onSessionStarted: (sessionId: string, systemType: string) => void;
}

const SYSTEMS = [
  { id: "A", label: "システム A", desc: "パターン A" },
  { id: "B", label: "システム B", desc: "パターン B" },
  { id: "C", label: "システム C", desc: "パターン C" },
  { id: "D", label: "システム D", desc: "パターン D" },
];

export default function LoginView({ onSessionStarted }: LoginViewProps) {
  // const [userId, setUserId] = useState<string>("");
  const { userId, setUserId } = useUserId()
  const [selectedSystem, setSelectedSystem] = useState<string>("A");
  const [loading, setLoading] = useState<boolean>(false);

  const handleStartSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId.trim() || !selectedSystem) return;

    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId.trim(),
          system_type: selectedSystem, // バックエンドへ送信
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "セッションの開始に失敗しました。");
      }

      const data = await res.json();
      onSessionStarted(data.session_id, selectedSystem);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col justify-center items-center flex-1 w-full p-4">
      <form onSubmit={handleStartSession} className="w-full max-w-md flex flex-col gap-5 p-6 border rounded-xl shadow-md bg-gray-50">
        <div>
          <label className="block font-semibold text-gray-700 mb-1">
            学籍番号を入力してください
          </label>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            className="w-full border p-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            placeholder="例: al21114"
            required
          />
        </div>

        <div>
          <label className="block font-semibold text-gray-700 mb-2">
            利用するシステムを選択してください
          </label>
          <div className="grid grid-cols-2 gap-2">
            {SYSTEMS.map((sys) => (
              <button
                key={sys.id}
                type="button"
                onClick={() => setSelectedSystem(sys.id)}
                className={`p-3 rounded-lg border text-center transition font-medium ${
                  selectedSystem === sys.id
                    ? "bg-blue-600 text-white border-blue-600 shadow-sm"
                    : "bg-white text-gray-700 border-gray-300 hover:bg-gray-100"
                }`}
              >
                {sys.label}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !userId.trim()}
          className="mt-2 bg-blue-600 text-white p-3 rounded-lg font-semibold hover:bg-blue-700 transition disabled:bg-gray-400"
        >
          {loading ? "セッション開始中..." : "対話を開始する"}
        </button>
      </form>
    </div>
  );
}