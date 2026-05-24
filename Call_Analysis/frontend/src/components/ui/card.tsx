"use client";

import * as React from "react";
import clsx from "clsx";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={clsx(
          "bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl p-8 shadow-xl",
          className
        )}
        {...props}
      />
    );
  }
);

Card.displayName = "Card";
