import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Header } from '@/components/header';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'MarketMind — AI Market Research',
  description: 'Agentic market research powered by LangGraph, pgvector, and OpenAI.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans bg-slate-50 min-h-screen">
        <Header />
        {children}
      </body>
    </html>
  );
}
