"use client";

import { useState } from "react";
import LoginView from "./components/LoginView";
import ChatField from "./components/ChatField";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentSystem, setCurrentSystem] = useState<string | null>(null);

  const handleSessionStarted = (sid: string, systemType: string) => {
    setSessionId(sid);
    setCurrentSystem(systemType);
  };

  const handleResetSession = () => {
    if (confirm("セッションを終了してシステム選択に戻りますか？")) {
      setSessionId(null);
      setCurrentSystem(null);
    }
  };

  return (
    <main className="flex flex-col h-screen w-full font-sans bg-white overflow-hidden p-3">
      <header className="py-3 border-b text-center shrink-0 flex justify-between items-center px-4">
        <div className="w-24"></div> {/* 左右バランス用スペース */}
        <h1 className="text-xl font-bold text-gray-800">
          対話型映画推薦システム
          {currentSystem && <span className="text-blue-500">(システム{currentSystem.toUpperCase()})</span>}
        </h1>
        <div className="text-right">
          {sessionId && (
            <button
              onClick={handleResetSession}
              className="border-2 p-3 font-bold bg-blue-200 hover:bg-blue-400"
            >
              システムを切り替える
            </button>
          )}
        </div>
      </header>

      {!sessionId ? (
        <div className="flex-1 flex items-center justify-center max-w-xl mx-auto w-full">
          <LoginView onSessionStarted={handleSessionStarted} />
        </div>
      ) : (
        <ChatField sessionId={sessionId} />
      )}
    </main>
  );
}