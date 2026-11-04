import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { UserIdProvider } from "./components/UserIdProvider";



export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
    >
      <UserIdProvider>
        <body className="min-h-full flex flex-col">{children}</body>
      </UserIdProvider>
    </html>
  );
}
