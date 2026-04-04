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

import Image from "next/image";
import { useEffect, useState, useCallback, useRef } from "react";
import { apiClient } from "@/lib/api";
import { WelcomeScreen } from "./welcome-screen";

interface SplashScreenProps {
  onComplete: () => void;
}

type AppState = "loading" | "welcome" | "ready";

export function SplashScreen({ onComplete }: SplashScreenProps) {
  const [appState, setAppState] = useState<AppState>("loading");
  const isMountedRef = useRef(true);

  const checkFirstLaunch = useCallback(async () => {
    try {
      const result = await apiClient.checkFirstLaunch();
      
      if (!isMountedRef.current) return;
      
      if (result.is_first_launch) {
        setAppState("welcome");
      } else {
        onComplete();
      }
    } catch (e) {
      if (!isMountedRef.current) return;
      console.error("Failed to check first launch:", e);
    }
  }, [onComplete]);

  useEffect(() => {
    isMountedRef.current = true;
    checkFirstLaunch();
    
    return () => {
      isMountedRef.current = false;
    };
  }, [checkFirstLaunch]);

  const handleWelcomeComplete = useCallback(() => {
    setAppState("ready");
    onComplete();
  }, [onComplete]);

  if (appState === "welcome") {
    return <WelcomeScreen onComplete={handleWelcomeComplete} />;
  }

  return (
    <div className="fixed inset-0 flex flex-col items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-8">
        <div className="relative w-48 h-20">
          <Image
            src="/load.svg"
            alt="Xi"
            fill
            className="object-contain"
            priority
          />
        </div>
      </div>
    </div>
  );
}
