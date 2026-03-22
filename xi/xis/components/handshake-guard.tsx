/**
 * Copyright © 2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of Xi.
 * The Xi project belongs to the Dunimd Team.
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
 */

"use client";

import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/lib/api";
import type { HandshakeResponse } from "@/types";

interface XisHandshakeGuardProps {
  children: React.ReactNode;
}

type HandshakeStatus = "idle" | "connecting" | "connected" | "failed";

export function XisHandshakeGuard({ children }: XisHandshakeGuardProps) {
  const [status, setStatus] = useState<HandshakeStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const performHandshake = useCallback(async () => {
    setStatus("connecting");
    setError(null);

    try {
      const response: HandshakeResponse = await apiClient.handshake();

      if (response.type === "handshake_ack") {
        setStatus("connected");
        setError(null);
      } else if (response.type === "handshake_error") {
        setStatus("failed");
        setError(response.error || "Handshake failed");
      }
    } catch (err) {
      setStatus("failed");
      setError(err instanceof Error ? err.message : "Connection to server failed");
    }
  }, [retryCount]);

  useEffect(() => {
    performHandshake();
  }, [performHandshake]);

  if (status === "connecting" || status === "idle") {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="text-sm text-muted-foreground">Connecting to Xi Server...</p>
        </div>
      </div>
    );
  }

  if (status === "failed") {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-6 max-w-md p-8">
          <div className="rounded-full bg-destructive/10 p-4">
            <svg
              className="h-12 w-12 text-destructive"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <div className="text-center">
            <h2 className="text-xl font-semibold text-foreground">Connection Failed</h2>
            <p className="mt-2 text-sm text-muted-foreground">{error}</p>
          </div>
          <div className="flex flex-col items-center gap-2 text-sm text-muted-foreground">
            <p>Make sure the Xi Server is running on port 3140</p>
            <p className="font-mono text-xs">http://127.0.0.1:3140</p>
          </div>
          <Button
            variant="secondary"
            onClick={() => {
              setRetryCount((c) => c + 1);
              setStatus("idle");
            }}
            className="mt-4"
          >
            Retry Connection
          </Button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
