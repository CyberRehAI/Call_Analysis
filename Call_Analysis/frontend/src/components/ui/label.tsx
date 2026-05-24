"use client";

import * as React from "react";
import clsx from "clsx";

export interface LabelProps
  extends React.LabelHTMLAttributes<HTMLLabelElement> {}

export const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={clsx("block text-sm font-medium text-white/80 mb-1", className)}
      {...props}
    />
  )
);

Label.displayName = "Label";
