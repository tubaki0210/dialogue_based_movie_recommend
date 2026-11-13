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
  const { userId, setUserId, selectedSystem, setSelectedSystem } = useUserId();
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
          system_type: selectedSystem,
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
    <div className="flex flex-col justify-center items-center flex-1 w-full p-4 overflow-y-auto">
      <form onSubmit={handleStartSession} className="w-full max-w-lg flex flex-col gap-5 p-6 border rounded-xl shadow-md bg-gray-50 my-auto">

        {/* --- 注意事項ボックス --- */}
        <div className="bg-amber-50 border border-amber-200 p-4 rounded-lg text-amber-900 space-y-2">
          <p className="font-bold text-amber-950 flex items-center gap-1">
            注意事項
          </p>
          <ul className="list-disc list-inside space-y-1 leading-relaxed">
            <li>各システム、<span className="font-semibold underline">推薦が2度(計6作品)行われるまで</span>対話を行ってください。</li>
            <li><span className="font-semibold text-red-600">システムA, B, Cでは、1度目の推薦が行われたら必ず「微妙です」と入力してください。</span></li>
            <li>システムA, B, Cでは、映画名や人名、年代（例: 「となりのトトロ」「新海誠」「2020年の映画」など）の好みの入力を禁止します。必ず「派手なアクション」「ファンタジー」等の内容に関する言葉を入力してください。</li>
            <li>システムごとに異なる好みを入力しても大丈夫です。</li>
            <li>対話終了後は対話履歴を見ながらアンケートに回答し、ホーム画面から次のシステムを選択してください。</li>
            <li>学籍番号と利用するシステムを選択し、対話を始めてください。</li>
          </ul>
        </div>

        <div>
          <label className="block font-semibold text-gray-700 mb-1 text-sm">
            学籍番号を入力してください
          </label>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            className="w-full border p-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white text-sm"
            placeholder="例: al21114"
            required
          />
        </div>

        <div>
          <label className="block font-semibold text-gray-700 mb-2 text-sm">
            利用するシステムを選択してください
          </label>
          <div className="grid grid-cols-2 gap-2">
            {SYSTEMS.map((sys) => (
              <button
                key={sys.id}
                type="button"
                onClick={() => setSelectedSystem(sys.id)}
                className={`p-3 rounded-lg border text-center transition font-medium text-sm ${
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
          disabled={loading || !userId.trim() || !selectedSystem}
          className="mt-2 bg-blue-600 text-white p-3 rounded-lg font-semibold hover:bg-blue-700 transition disabled:bg-gray-400 text-sm"
        >
          {loading ? "セッション開始中..." : "対話を開始する"}
        </button>
      </form>
    </div>
  );
}