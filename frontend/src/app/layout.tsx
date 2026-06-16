import type { Metadata } from 'next';
import { Playfair_Display, DM_Sans, IBM_Plex_Mono } from 'next/font/google';
import './globals.css';
import { Header } from '@/components/header';

const playfair = Playfair_Display({
  subsets: ['latin'],
  weight: ['600', '700', '800'],
  variable: '--font-display',
  display: 'swap',
});

const dmSans = DM_Sans({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600'],
  variable: '--font-body',
  display: 'swap',
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'MarketMind — AI Market Research',
  description: 'Agentic market research powered by LangGraph, pgvector, and OpenAI.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${playfair.variable} ${dmSans.variable} ${ibmPlexMono.variable}`}>
      <body className="font-body bg-bg-base min-h-screen text-ink-secondary antialiased">
        <Header />
        {children}
      </body>
    </html>
  );
}
