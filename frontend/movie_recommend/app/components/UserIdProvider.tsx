"use client";
import React, { createContext, useState, Dispatch, SetStateAction, useContext } from "react";

// 1. 型定義に setUserId を追加する
interface UserIdContextType {
  userId: string;
  setUserId: Dispatch<SetStateAction<string>>;
  selectedSystem: string;
  setSelectedSystem: Dispatch<SetStateAction<string>>;
}

export const UserIdContext = createContext<UserIdContextType | null>(null);

export const UserIdProvider = ({ children }: { children: React.ReactNode }) => {
  const [userId, setUserId] = useState("");
  const [selectedSystem, setSelectedSystem] = useState<string>("A");

  // 2. value に setUserId を含める
  const value: UserIdContextType = {
    userId,
    setUserId,
    selectedSystem,
    setSelectedSystem
  };

  return <UserIdContext.Provider value={value}>{children}</UserIdContext.Provider>;
};

export const useUserId = () => {
    const context = useContext(UserIdContext)
    if (context === null) {
        throw new Error("error")
    }
    return context
}