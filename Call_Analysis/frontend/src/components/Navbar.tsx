'use client';

import Link from 'next/link';
import { BarChart3, History, Info } from 'lucide-react';

const navItems = [
  { href: '#home', label: 'Home', icon: BarChart3 },
  { href: '#history', label: 'History', icon: History },
  { href: '#results', label: 'Results', icon: BarChart3 },
  { href: '#about', label: 'About', icon: Info },
];

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/95 border-b border-slate-700/50 backdrop-blur-sm">
      <div className="container mx-auto px-6">
        <div className="flex justify-between items-center h-14">
          <Link href="#home" className="flex items-center gap-2 text-slate-100 font-semibold hover:text-white transition-colors">
            <div className="w-8 h-8 rounded-lg bg-teal-500/20 flex items-center justify-center">
              <BarChart3 className="h-4 w-4 text-teal-400" />
            </div>
            <span>Call Analysis</span>
          </Link>
          <div className="flex items-center gap-1">
            {navItems.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                scroll
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
