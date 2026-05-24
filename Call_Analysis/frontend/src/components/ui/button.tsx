"use client";

import * as React from "react";
import clsx from "clsx";

export type ButtonVariant = "primary" | "secondary" | "ghost";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const baseClasses =
  "inline-flex items-center justify-center rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:opacity-50 disabled:pointer-events-none gap-2";

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-gradient-to-r from-blue-900 to-slate-950 hover:from-black hover:to-slate-900 text-white shadow-lg hover:shadow-xl", 
  secondary:
    "bg-white/10 hover:bg-white/20 border border-white/20 hover:border-white/30 text-white backdrop-blur-sm", 
  ghost:
    "bg-transparent hover:bg-white/10 text-white/80 hover:text-white",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={clsx(baseClasses, variantClasses[variant], className)}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
