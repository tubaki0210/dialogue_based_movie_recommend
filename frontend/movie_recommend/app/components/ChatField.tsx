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

  // 【追加】タイピング演出用の文字キューと通信完了フラグ
  const queueRef = useRef<string[]>([]);
  const isStreamDoneRef = useRef<boolean>(false);

  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  useEffect(() => {
    if (!loading) {
      inputRef.current?.focus()
    }
  },[loading])

  // 【追加】タイピング描画ループ（15msごとにキューから1文字ずつ画面に追加）
  useEffect(() => {
    const timer = setInterval(() => {
      if (queueRef.current.length > 0) {
        const nextChar = queueRef.current.shift();
        if (nextChar !== undefined) {
          setMessages((prev) => {
            const newMsgs = [...prev];
            const lastIdx = newMsgs.length - 1;
            if (lastIdx < 0) return prev;
            newMsgs[lastIdx] = {
              ...newMsgs[lastIdx],
              content: newMsgs[lastIdx].content + nextChar,
            };
            return newMsgs;
          });
        }
      } else if (isStreamDoneRef.current && loading) {
        // 通信が完了し、かつキューの文字をすべて表示し切ったら完了状態にする
        setLoading(false);
      }
    }, 15); // 描画速度（15ms ≒ 1文字あたり約0.015秒）。お好みで調整可能です

    return () => clearInterval(timer);
  }, [loading]);

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


    // キューと完了フラグの初期化
    queueRef.current = [];
    isStreamDoneRef.current = false;
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
                // 【変更】直接 messages に足さず、1文字ずつバラしてキューに追加する
                queueRef.current.push(...parsed.content.split(""));
              }
            } catch (jsonErr) {}
          }
        }
      }
    } catch (error) {
      console.error(error);
      alert("通信エラーが発生しました。");
      setLoading(false);
    } finally {
      // API通信自体が終わったことを記録（キューの文字がすべて吐き出されるとタイマー側で loading が false になります）
      isStreamDoneRef.current = true;
    }
  };

  return (
<div className="flex flex-col flex-1 w-full overflow-hidden bg-gray-50">
      
      {/* メッセージ一覧エリア（ここだけが縦スクロールする） */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* 大画面でも横に広がりすぎないよう、中身を中央寄せしつつ最大幅を制限 */}
        <div className="max-w-4xl mx-auto w-full space-y-4">
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
                  <span className="animate-pulse text-gray-400">システムが応答を生成中...</span>
                ) : (
                  ""
                ))}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* 入力フォームエリア（画面下部に固定） */}
      <div className="bg-white border-t p-3 shrink-0">
        <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto flex gap-2">
          <input
            type="text"
            value={input}
            ref={inputRef}
            onChange={(e) => setInput(e.target.value)}
            placeholder="好みや嫌いな要素を入力"
            disabled={loading}
            className="flex-1 border rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 text-white px-5 py-2.5 rounded-lg text-sm font-semibold hover:bg-blue-700 transition disabled:bg-gray-300"
          >
            送信
          </button>
        </form>
      </div>

    </div>
  );
}