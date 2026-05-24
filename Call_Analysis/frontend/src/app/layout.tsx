import type { Metadata } from 'next';
import { DM_Sans } from 'next/font/google';
import './globals.css';
import { Toaster } from 'react-hot-toast';
import Navbar from '@/components/Navbar';

const dmSans = DM_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-dm-sans',
});

export const metadata: Metadata = {
  title: 'Call Analysis System | Conversation Intelligence',
  description: 'Sentiment and emotion analysis for phone calls with actionable insights.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={dmSans.className}>
        <div className="min-h-screen bg-slate-900">
          <Navbar />
          <main className="relative">
            {children}
          </main>
          <Toaster
            position="bottom-right"
            toastOptions={{
              duration: 4000,
              style: { background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155' },
              success: { iconTheme: { primary: '#0d9488', secondary: '#fff' } },
              error: { iconTheme: { primary: '#ef4444', secondary: '#fff' } },
            }}
          />
        </div>
      </body>
    </html>
  );
}
