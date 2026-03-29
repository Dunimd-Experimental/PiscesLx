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

import { useEffect, useState } from "react";
import { Shield, Server, Globe, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiClient } from "@/lib/api";

type ApiAccessStatus = "idle" | "checking" | "accessible" | "unavailable";

interface ApiAccessInfo {
  status: ApiAccessStatus;
  baseUrl: string;
  version?: string;
  endpoints?: string[];
  error?: string;
}

interface XisApiAccessGuardProps {
  children: React.ReactNode;
}

export function XisApiAccessGuard({ children }: XisApiAccessGuardProps) {
  const [accessInfo, setAccessInfo] = useState<ApiAccessInfo>({
    status: "idle",
    baseUrl: "",
  });

  useEffect(() => {
    checkApiAccess();
  }, []);

  const checkApiAccess = async () => {
    setAccessInfo((prev) => ({ ...prev, status: "checking" }));

    try {
      const health = await apiClient.healthCheck();
      const config = await apiClient.getConfig();

      const configData = config as Record<string, unknown>;
      const projectData = configData?.project as Record<string, unknown> | undefined;
      
      setAccessInfo({
        status: "accessible",
        baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:3140",
        version: (projectData?.version as string) || "unknown",
        endpoints: [
          "/v1/xi/config",
          "/v1/xi/paths",
          "/v1/xi/commands",
          "/v1/runs",
          "/healthz",
        ],
      });
    } catch (err) {
      setAccessInfo({
        status: "unavailable",
        baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:3140",
        error: err instanceof Error ? err.message : "API is not accessible",
      });
    }
  };

  if (accessInfo.status === "checking" || accessInfo.status === "idle") {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="text-sm text-muted-foreground">Checking API access...</p>
        </div>
      </div>
    );
  }

  if (accessInfo.status === "unavailable") {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background p-4">
        <Card className="max-w-lg w-full">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10">
              <Server className="h-8 w-8 text-destructive" />
            </div>
            <CardTitle className="text-xl">Backend API Unavailable</CardTitle>
            <CardDescription>
              Unable to connect to the Xi backend server
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="rounded-lg bg-muted p-4">
              <div className="flex items-center gap-3">
                <Globe className="h-5 w-5 text-muted-foreground" />
                <div className="flex-1">
                  <p className="text-sm font-medium">Server URL</p>
                  <p className="font-mono text-xs text-muted-foreground">{accessInfo.baseUrl}</p>
                </div>
              </div>
            </div>

            {accessInfo.error && (
              <div className="rounded-lg bg-destructive/5 p-4">
                <p className="text-sm text-destructive">{accessInfo.error}</p>
              </div>
            )}

            <div className="space-y-2 text-sm text-muted-foreground">
              <p className="font-medium text-foreground">Troubleshooting:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Ensure the Xi Server is running</li>
                <li>Check if the server port (3140) is accessible</li>
                <li>Verify firewall settings</li>
                <li>Check server logs for errors</li>
              </ul>
            </div>

            <div className="flex items-center gap-2 rounded-lg bg-yellow-500/5 p-4 text-sm text-yellow-600">
              <Lock className="h-4 w-4 flex-shrink-0" />
              <p>This interface requires a working backend connection to function properly.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <>{children}</>;
}

export function ApiAccessPanel() {
  const [accessInfo, setAccessInfo] = useState<ApiAccessInfo>({
    status: "idle",
    baseUrl: "",
  });

  useEffect(() => {
    checkApiAccess();
  }, []);

  const checkApiAccess = async () => {
    setAccessInfo((prev) => ({ ...prev, status: "checking" }));

    try {
      const health = await apiClient.healthCheck();
      const config = await apiClient.getConfig();
      
      const configData = config as Record<string, unknown>;
      const projectData = configData?.project as Record<string, unknown> | undefined;

      setAccessInfo({
        status: "accessible",
        baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:3140",
        version: (projectData?.version as string) || "unknown",
        endpoints: [
          "/v1/xi/config",
          "/v1/xi/paths",
          "/v1/xi/commands",
          "/v1/runs",
          "/healthz",
        ],
      });
    } catch (err) {
      setAccessInfo({
        status: "unavailable",
        baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:3140",
        error: err instanceof Error ? err.message : "API is not accessible",
      });
    }
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className={`flex h-10 w-10 items-center justify-center rounded-full ${
            accessInfo.status === "accessible"
              ? "bg-green-500/10"
              : accessInfo.status === "unavailable"
              ? "bg-destructive/10"
              : "bg-muted"
          }`}>
            {accessInfo.status === "accessible" ? (
              <Shield className="h-5 w-5 text-green-500" />
            ) : accessInfo.status === "unavailable" ? (
              <Server className="h-5 w-5 text-destructive" />
            ) : (
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            )}
          </div>
          <div>
            <CardTitle className="text-base">Backend API Status</CardTitle>
            <CardDescription>
              {accessInfo.status === "accessible"
                ? "Connected and operational"
                : accessInfo.status === "unavailable"
                ? "Connection failed"
                : "Checking connection..."}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {accessInfo.status === "accessible" && (
          <>
            <div className="space-y-2">
              <p className="text-sm font-medium">Server URL</p>
              <p className="font-mono text-xs text-muted-foreground bg-muted p-2 rounded">
                {accessInfo.baseUrl}
              </p>
            </div>
            {accessInfo.version && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Version</p>
                <p className="text-sm text-muted-foreground">{accessInfo.version}</p>
              </div>
            )}
            {accessInfo.endpoints && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Available Endpoints</p>
                <div className="flex flex-wrap gap-1">
                  {accessInfo.endpoints.map((endpoint) => (
                    <span
                      key={endpoint}
                      className="inline-flex items-center rounded-md bg-muted px-2 py-1 text-xs font-medium"
                    >
                      {endpoint}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {accessInfo.status === "unavailable" && accessInfo.error && (
          <div className="rounded-lg bg-destructive/5 p-3">
            <p className="text-sm text-destructive">{accessInfo.error}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
