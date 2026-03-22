/**
 * Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of PiscesL1.
 * The Pisces L1 project belongs to the Dunimd Team.
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

import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Loader2, AlertCircle, CheckCircle2, ArrowRight } from "lucide-react";
import { apiClient } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface WelcomeScreenProps {
  onComplete: () => void;
}

const AGREEMENT_TEXT = `Xi Studio Service Agreement

Effective Date: March 21, 2026

1. ACCEPTANCE OF TERMS

By clicking "Agree" and using Xi Studio ("the Software"), you agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the Software.

2. DESCRIPTION OF SERVICE

Xi Studio is an AI workstation platform that provides tools for model training, inference, and data management. The Software is provided for personal and commercial use subject to these terms.

3. USER RESPONSIBILITIES

You are responsible for:
- Maintaining the security of your account and data
- Ensuring compliance with applicable AI regulations in your jurisdiction
- Proper use of AI models and generated content
- All activities that occur under your account

4. ACCEPTABLE USE

You agree NOT to:
- Use the Software for any illegal or unauthorized purpose
- Violate any applicable local, state, national, or international law
- Infringe upon the intellectual property rights of others
- Attempt to reverse engineer, decompile, or disassemble the Software
- Use the Software to create harmful, malicious, or unethical AI content

5. AI REGULATORY COMPLIANCE

Users must comply with all applicable AI regulations, including but not limited to:
- Content labeling and disclosure requirements
- Data privacy and protection regulations
- AI transparency and accountability requirements
- Prohibited use cases as defined by applicable law

6. INTELLECTUAL PROPERTY

The Software and its original content, features, and functionality are owned by Wenze Wei and the Dunimd Team. You retain ownership of your data and content created using the Software.

7. DISCLAIMER

THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

By clicking "Agree", you acknowledge that you have read, understood, and agree to be bound by these Terms of Service.`;

interface ValidationStep {
  step: string;
  message: string;
  status: "pending" | "checking" | "done";
  valid: boolean;
  error: string | null;
  data: Record<string, unknown> | null;
}

const STEP_LABELS: Record<string, string> = {
  xi_toml_syntax: "Xi Configuration Syntax",
  project_info: "Project Information",
  subcommands: "Subcommand Configurations",
  python_env: "Python Environment",
  ui_config: "UI Configuration",
  venv_create: "Virtual Environment",
  install_deps: "Installing Dependencies",
  verify_setup: "Verifying Setup",
};

const SETUP_STEP_LABELS: Record<string, string> = {
  venv_create: "Virtual Environment",
  install_deps: "Installing Dependencies",
  verify_setup: "Verifying Setup",
};

export function WelcomeScreen({ onComplete }: WelcomeScreenProps) {
  const [step, setStep] = useState<0 | 1 | 2 | 3 | 4>(0);
  const [countdown, setCountdown] = useState(5);
  const [steps, setSteps] = useState<ValidationStep[]>([]);
  const [setupSteps, setSetupSteps] = useState<ValidationStep[]>([]);
  const [currentMessage, setCurrentMessage] = useState<string>("Initializing validation...");
  const [setupCurrentMessage, setSetupCurrentMessage] = useState<string>("Initializing environment setup...");
  const [allPassed, setAllPassed] = useState(false);
  const [setupAllPassed, setSetupAllPassed] = useState(false);
  const [showDisagreeDialog, setShowDisagreeDialog] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [setupError, setSetupError] = useState<string | null>(null);

  useEffect(() => {
    if (step === 1 && countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(countdown - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [step, countdown]);

  const runConfigValidation = useCallback(async () => {
    setSteps([]);
    setCurrentMessage("Validating configuration...");
    setAllPassed(false);
    setValidationError(null);

    try {
      const response = await fetch("/v1/xi/validate-config", {
        headers: {
          "Accept": "text/event-stream",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.event === "checking") {
                setCurrentMessage(data.message || `Checking ${STEP_LABELS[data.step] || data.step}...`);
                setSteps((prev) => {
                  const existing = prev.find((s) => s.step === data.step);
                  if (existing) {
                    return prev.map((s) =>
                      s.step === data.step ? { ...s, status: "checking", message: data.message } : s
                    );
                  }
                  return [
                    ...prev,
                    {
                      step: data.step,
                      message: data.message || "",
                      status: "checking" as const,
                      valid: false,
                      error: null,
                      data: null,
                    },
                  ];
                });
              } else if (data.event === "result") {
                setSteps((prev) =>
                  prev.map((s) =>
                    s.step === data.step
                      ? {
                          step: s.step,
                          message: s.message,
                          status: "done" as const,
                          valid: data.valid,
                          error: data.error,
                          data: data.data || null,
                        }
                      : s
                  )
                );
                if (!data.valid && data.error) {
                  setCurrentMessage(`Error: ${data.error}`);
                }
              } else if (data.event === "done") {
                setAllPassed(data.valid);
                if (data.valid) {
                  setCurrentMessage("All validations passed!");
                  setTimeout(() => setStep(3), 500);
                } else {
                  setCurrentMessage("Validation failed. Please check the errors above.");
                }
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Validation error:", error);
      setValidationError(`Failed to connect: ${error instanceof Error ? error.message : "Unknown error"}. Please ensure the backend server is running.`);
      setCurrentMessage("Connection failed");
    }
  }, []);

  useEffect(() => {
    if (step === 2) {
      runConfigValidation();
    }
  }, [step, runConfigValidation]);

  const handleStart = () => {
    setStep(1);
  };

  const handleAgree = () => {
    if (countdown === 0) {
      setStep(3);
    }
  };

  const runSetupEnvironment = useCallback(async () => {
    setSetupSteps([]);
    setSetupCurrentMessage("Initializing environment setup...");
    setSetupAllPassed(false);
    setSetupError(null);

    try {
      const response = await fetch("/v1/xi/setup-environment", {
        headers: {
          "Accept": "text/event-stream",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.event === "checking") {
                setSetupCurrentMessage(data.message || `Setting up ${SETUP_STEP_LABELS[data.step] || data.step}...`);
                setSetupSteps((prev) => {
                  const existing = prev.find((s) => s.step === data.step);
                  if (existing) {
                    return prev.map((s) =>
                      s.step === data.step ? { ...s, status: "checking", message: data.message } : s
                    );
                  }
                  return [
                    ...prev,
                    {
                      step: data.step,
                      message: data.message || "",
                      status: "checking" as const,
                      valid: false,
                      error: null,
                      data: null,
                    },
                  ];
                });
              } else if (data.event === "result") {
                setSetupSteps((prev) =>
                  prev.map((s) =>
                    s.step === data.step
                      ? {
                          step: s.step,
                          message: s.message,
                          status: "done" as const,
                          valid: data.valid,
                          error: data.error,
                          data: data.data || null,
                        }
                      : s
                  )
                );
                if (!data.valid && data.error) {
                  setSetupCurrentMessage(`Error: ${data.error}`);
                }
              } else if (data.event === "done") {
                setSetupAllPassed(data.valid);
                if (data.valid) {
                  setSetupCurrentMessage("Environment setup complete!");
                  setTimeout(() => setStep(4), 500);
                } else {
                  setSetupCurrentMessage("Environment setup failed. Please check the errors above.");
                }
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Setup error:", error);
      setSetupError(`Failed to connect: ${error instanceof Error ? error.message : "Unknown error"}. Please ensure the backend server is running.`);
      setSetupCurrentMessage("Connection failed");
    }
  }, []);

  useEffect(() => {
    if (step === 2) {
      runConfigValidation();
    }
  }, [step, runConfigValidation]);

  useEffect(() => {
    if (step === 3) {
      runSetupEnvironment();
    }
  }, [step, runSetupEnvironment]);

  const handleSetupRetry = () => {
    setStep(3);
  };

  const handleDisagree = () => {
    setShowDisagreeDialog(true);
  };

  const handleDisagreeConfirm = () => {
    setShowDisagreeDialog(false);
    setStep(0);
    setCountdown(5);
  };

  const handleDisagreeCancel = () => {
    setShowDisagreeDialog(false);
  };

  const handleComplete = async () => {
    try {
      await apiClient.completeFirstLaunch();
    } catch (e) {
      console.error("Failed to complete first launch:", e);
    }
    onComplete();
  };

  const handleRetry = () => {
    setStep(2);
  };

  const getStatusIcon = (s: ValidationStep) => {
    if (s.status === "checking") {
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    }
    if (s.status === "pending") {
      return <div className="h-4 w-4 rounded-full border-2 border-muted-foreground/30" />;
    }
    if (s.valid) {
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    }
    return <AlertCircle className="h-4 w-4 text-red-500" />;
  };

  if (step === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-background px-8">
        <div className="w-full max-w-2xl text-center">
          <div className="mb-8 flex justify-center">
            <Image
              src="/xi-logo.svg"
              alt="Xi Logo"
              width={200}
              height={80}
              priority
            />
          </div>

          <h1 className="text-4xl font-bold mb-10 text-foreground">
            Welcome to Xi Studio
          </h1>

          <Button
            variant="outline"
            onClick={handleStart}
            size="lg"
            className="px-8"
          >
            Get Started
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <>
        <div className="flex flex-col items-center justify-center min-h-screen bg-background px-8">
          <div className="w-full max-w-2xl">
            <h2 className="text-xl font-semibold mb-4 text-center">Service Agreement</h2>
            
            <div className="border border-border rounded-lg overflow-hidden">
              <div className="p-6 h-[300px] overflow-y-auto bg-muted/30 text-sm whitespace-pre-wrap textarea--agreement">
                {AGREEMENT_TEXT}
              </div>
            </div>

            <div className="flex justify-center gap-4 mt-8">
              <Button
                variant="outline"
                onClick={handleDisagree}
                size="lg"
                className="px-8"
              >
                Disagree
              </Button>
              <Button
                variant="outline"
                onClick={handleAgree}
                disabled={countdown > 0}
                size="lg"
                className="px-8"
              >
                {countdown > 0 ? `Agree (${countdown}s)` : "Agree"}
              </Button>
            </div>
          </div>
        </div>

        <Dialog open={showDisagreeDialog} onOpenChange={setShowDisagreeDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Decline Agreement</DialogTitle>
              <DialogDescription>
                You must agree to the Service Agreement to use Xi Studio. 
                If you decline, you will be returned to the welcome screen.
                Are you sure you want to decline?
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={handleDisagreeCancel}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDisagreeConfirm}>
                Decline
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </>
    );
  }

  const isValidationComplete = allPassed && steps.length > 0 && steps.every(s => s.status === "done");
  const hasValidationFailed = steps.some(s => s.status === "done" && !s.valid);

  const isSetupComplete = setupAllPassed && setupSteps.length > 0 && setupSteps.every(s => s.status === "done");
  const hasSetupFailed = setupSteps.some(s => s.status === "done" && !s.valid);

  if (step === 2) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-background px-8">
        <div className="w-full max-w-lg">
          <h2 className="text-xl font-semibold mb-8 text-center">Validating Configuration</h2>

          {!isValidationComplete && !hasValidationFailed && (
            <div className="flex flex-col items-center justify-center gap-6 py-8">
              <div className="relative w-32 h-14">
                <Image
                  src="/load.svg"
                  alt="Loading"
                  fill
                  className="object-contain animate-pulse"
                  priority
                />
              </div>
              {currentMessage && (
                <p className="text-sm text-muted-foreground text-center">{currentMessage}</p>
              )}
            </div>
          )}

          <div className="space-y-2">
            {steps.filter(s => s.status !== "pending").map((s) => (
              <div
                key={s.step}
                className={`flex items-center gap-4 p-3 rounded-lg border ${
                  s.status === "checking"
                    ? "border-primary"
                    : s.valid
                      ? "border-green-500"
                      : "border-red-500"
                }`}
              >
                <div className="flex-shrink-0">{getStatusIcon(s)}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    {STEP_LABELS[s.step] || s.step}
                  </div>
                  {s.status === "done" && s.error && (
                    <div className={`text-xs mt-0.5 ${s.valid ? "text-yellow-600" : "text-red-500"}`}>
                      {s.error}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {validationError && (
            <div className="mt-6 flex justify-center">
              <Button
                variant="outline"
                onClick={handleRetry}
                size="lg"
                className="px-8"
              >
                Retry
              </Button>
            </div>
          )}

          {hasValidationFailed && (
            <div className="mt-6 flex justify-center">
              <Button
                variant="outline"
                onClick={handleRetry}
                size="lg"
                className="px-8"
              >
                Retry
              </Button>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (step === 3) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-background px-8">
        <div className="w-full max-w-lg">
          <h2 className="text-xl font-semibold mb-8 text-center">Environment Setup</h2>

          {!isSetupComplete && !hasSetupFailed && (
            <div className="flex flex-col items-center justify-center gap-6 py-8">
              <div className="relative w-32 h-14">
                <Image
                  src="/load.svg"
                  alt="Loading"
                  fill
                  className="object-contain animate-pulse"
                  priority
                />
              </div>
              {setupCurrentMessage && (
                <p className="text-sm text-muted-foreground text-center">{setupCurrentMessage}</p>
              )}
            </div>
          )}

          <div className="space-y-2">
            {setupSteps.filter(s => s.status !== "pending").map((s) => (
              <div
                key={s.step}
                className={`flex items-center gap-4 p-3 rounded-lg border ${
                  s.status === "checking"
                    ? "border-primary"
                    : s.valid
                      ? "border-green-500"
                      : "border-red-500"
                }`}
              >
                <div className="flex-shrink-0">{getStatusIcon(s)}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    {SETUP_STEP_LABELS[s.step] || s.step}
                  </div>
                  {s.status === "done" && s.error && (
                    <div className={`text-xs mt-0.5 ${s.valid ? "text-yellow-600" : "text-red-500"}`}>
                      {s.error}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {setupError && (
            <div className="mt-6 flex justify-center">
              <Button
                variant="outline"
                onClick={handleSetupRetry}
                size="lg"
                className="px-8"
              >
                Retry
              </Button>
            </div>
          )}

          {hasSetupFailed && (
            <div className="mt-6 flex justify-center">
              <Button
                variant="outline"
                onClick={handleSetupRetry}
                size="lg"
                className="px-8"
              >
                Retry
              </Button>
            </div>
          )}

          {isSetupComplete && (
            <div className="mt-6 flex justify-center">
              <p className="text-sm text-muted-foreground">Environment setup complete. Loading...</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (step === 4) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-background px-8">
        <div className="w-full max-w-2xl text-center">
          <div className="mb-8 flex justify-center">
            <Image
              src="/xi-logo.svg"
              alt="Xi Logo"
              width={200}
              height={80}
              priority
            />
          </div>

          <div className="flex justify-center mb-8">
            <CheckCircle2 className="h-24 w-24 text-green-500" />
          </div>

          <h1 className="text-4xl font-bold mb-4 text-foreground">
            Congratulations!
          </h1>

          <p className="text-xl text-muted-foreground mb-10">
            You can now use Xi Studio
          </p>

          <Button
            variant="outline"
            onClick={handleComplete}
            size="lg"
            className="px-8"
          >
            Get Started
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    );
  }
}