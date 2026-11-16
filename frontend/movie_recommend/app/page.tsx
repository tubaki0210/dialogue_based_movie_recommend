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
    <main className="flex flex-col h-screen w-full font-sans bg-white overflow-hidden p-2 sm:p-3">
      <header className="py-2.5 sm:py-3 border-b text-center shrink-0 flex flex-col sm:flex-row justify-between items-center px-2 sm:px-4 gap-2 sm:gap-0">
        {/* スマホでは非表示 or 小さく、PCではバランス用スペース */}
        <div className="hidden sm:block w-24"></div> 
        
        <h1 className="text-base sm:text-xl font-bold text-gray-800 leading-tight">
          対話型映画推薦システム
          {currentSystem && <span className="text-blue-500 ml-1">(システム{currentSystem.toUpperCase()})</span>}
        </h1>

        <div className="text-right w-full sm:w-auto flex justify-center sm:justify-end">
          {sessionId && (
            <button
              onClick={handleResetSession}
              className="border-2 border-blue-300 px-3 py-1.5 sm:p-3 text-xs sm:text-sm font-bold bg-blue-200 hover:bg-blue-400 rounded-xl sm:rounded-2xl transition"
            >
              システムを切り替える
            </button>
          )}
        </div>
      </header>

      {!sessionId ? (
        <div className="flex-1 flex items-center justify-center max-w-xl mx-auto w-full p-2">
          <LoginView onSessionStarted={handleSessionStarted} />
        </div>
      ) : (
        <ChatField sessionId={sessionId} />
      )}
    </main>
  );
}