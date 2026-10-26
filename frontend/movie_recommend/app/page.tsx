"use client";

import { useState } from "react";
import LoginView from "./components/LoginView";
import ChatField from "./components/ChatField";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [initialMessage, setInitialMessage] = useState<string>("");

  return (
<main className="flex flex-col h-screen w-full font-sans bg-white overflow-hidden">
      <header className="py-3 border-b text-center shrink-0">
        <h1 className="text-xl font-bold text-gray-800">対話型映画推薦システム</h1>
      </header>
      
      {!sessionId ? (
        <div className="flex-1 flex items-center justify-center">
          <LoginView
            onSessionStarted={(sid) => {
              setSessionId(sid);
            }}
          />
        </div>
      ) : (
        <ChatField sessionId={sessionId} initialMessage={initialMessage} />
      )}
    </main>
  );
}