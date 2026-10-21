"use client";

import { useState } from "react";
import LoginView from "./components/LoginView";
import ChatField from "./components/ChatField";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);

  return (
    <main className="flex flex-col h-screen max-w-6xl mx-auto p-4 font-sans bg-white">
      <header className="py-2 border-b mb-4">
        <h1 className="text-xl font-bold text-center text-gray-800">対話型映画推薦システム</h1>
      </header>
      {!sessionId ? (
        <LoginView
          onSessionStarted={(sid) => {
            setSessionId(sid);
          }}
        />
      ) : (
        <ChatField sessionId={sessionId} />
      )}
    </main>
  );
}