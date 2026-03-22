/**
 * Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of PiscesL1.
 * The PiscesL1 project belongs to the Dunimd Team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * DISCLAIMER: Users must comply with applicable AI regulations.
 * Non-compliance may result in service termination or legal liability.
 */

"use client";

import { Card, CardContent } from "@/components/ui/card";
import { AlertCircle, XCircle, Ban } from "lucide-react";

interface UnavailableStateProps {
  title?: string;
  reason?: string;
  variant?: "error" | "warning" | "muted";
  size?: "sm" | "md" | "lg";
  showIcon?: boolean;
}

export function UnavailableState({
  title = "Unavailable",
  reason,
  variant = "muted",
  size = "md",
  showIcon = true,
}: UnavailableStateProps) {
  const iconSize = size === "sm" ? "h-6 w-6" : size === "lg" ? "h-16 w-16" : "h-12 w-12";
  const textSize = size === "sm" ? "text-sm" : size === "lg" ? "text-xl" : "text-lg";
  const descSize = size === "sm" ? "text-xs" : size === "lg" ? "text-base" : "text-sm";
  const padding = size === "sm" ? "py-6" : size === "lg" ? "py-16" : "py-12";

  const Icon = variant === "error" ? XCircle : variant === "warning" ? AlertCircle : Ban;
  const iconColor = variant === "error" 
    ? "text-destructive" 
    : variant === "warning" 
    ? "text-yellow-500" 
    : "text-muted-foreground";

  const borderColor = variant === "error" 
    ? "border-destructive/50 bg-destructive/5" 
    : variant === "warning" 
    ? "border-yellow-500/50 bg-yellow-500/5" 
    : "border-dashed";

  return (
    <Card className={borderColor}>
      <CardContent className={`flex flex-col items-center justify-center ${padding}`}>
        {showIcon && <Icon className={`${iconSize} ${iconColor} mb-4`} />}
        <h3 className={`${textSize} font-medium mb-2`}>{title}</h3>
        {reason && (
          <p className={`${descSize} text-muted-foreground text-center max-w-md`}>
            {reason}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

interface UnavailableFieldProps {
  reason?: string;
  compact?: boolean;
}

export function UnavailableField({ reason = "Not available", compact = false }: UnavailableFieldProps) {
  if (compact) {
    return (
      <span className="text-xs text-muted-foreground italic">{reason}</span>
    );
  }

  return (
    <div className="flex items-center gap-2 p-3 rounded-md bg-muted/50 border border-dashed">
      <XCircle className="h-4 w-4 text-muted-foreground flex-shrink-0" />
      <span className="text-sm text-muted-foreground">{reason}</span>
    </div>
  );
}

interface UnavailableTabProps {
  reason?: string;
}

export function UnavailableTab({ reason }: UnavailableTabProps) {
  return (
    <UnavailableState
      title="Tab Unavailable"
      reason={reason || "This tab is not available due to missing configuration."}
      variant="muted"
      size="md"
    />
  );
}

interface UnavailableCommandProps {
  command: string;
  reason?: string;
}

export function UnavailableCommand({ command, reason }: UnavailableCommandProps) {
  return (
    <UnavailableState
      title={`${command.charAt(0).toUpperCase() + command.slice(1)} Unavailable`}
      reason={reason || `The ${command} command is not available due to missing configuration.`}
      variant="error"
      size="lg"
    />
  );
}
