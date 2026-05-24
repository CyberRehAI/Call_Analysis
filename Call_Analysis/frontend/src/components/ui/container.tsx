"use client";

import * as React from "react";
import clsx from "clsx";

export interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Container = React.forwardRef<HTMLDivElement, ContainerProps>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={clsx("max-w-6xl mx-auto px-6 w-full", className)}
      {...props}
    />
  )
);

Container.displayName = "Container";
