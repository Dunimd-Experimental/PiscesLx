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

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Loader2, AlertCircle, CheckCircle2, ArrowRight, ChevronDown, ChevronRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { GlobalOverlay } from "@/components/layout/global-overlay";
import { PiscesL1WelcomeWS, WelcomeServerMessage } from "@/lib/api/welcome-ws";

interface WelcomeScreenProps {
  onComplete: () => void;
}

interface ValidationStep {
  step: string;
  message: string;
  status: "pending" | "checking" | "done";
  valid: boolean;
  error: string | null;
  data: Record<string, unknown> | null;
  warnings?: string[];
}

const STEP_LABELS: Record<string, string> = {
  xi_toml_syntax: "Xi Configuration Syntax",
  project_info: "Project Information",
  subcommands: "Subcommand Configurations",
  python_env: "Python Environment",
  ui_config: "UI Configuration",
  paths_config: "Paths Configuration",
};

const SETUP_STEP_LABELS: Record<string, string> = {
  venv_create: "Virtual Environment",
  install_deps: "Installing Dependencies",
  create_dirs: "Creating Directories",
  verify_setup: "Verifying Setup",
};

function ValidationStepItem({
  step,
  labels,
  expandedSteps,
  toggleStep
}: {
  step: ValidationStep;
  labels: Record<string, string>;
  expandedSteps: Set<string>;
  toggleStep: (stepId: string) => void;
}) {
  const isExpanded = expandedSteps.has(step.step);
  const hasDetails = step.data || step.error || (step.warnings && step.warnings.length > 0);
  const label = labels[step.step] || step.step;

  const getStatusIcon = () => {
    if (step.status === "checking") {
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    }
    if (step.status === "pending") {
      return <div className="h-4 w-4 rounded-full border-2 border-muted-foreground/30" />;
    }
    if (step.valid) {
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    }
    return <AlertCircle className="h-4 w-4 text-red-500" />;
  };

  const renderDataValue = (value: unknown, depth: number = 0): React.ReactNode => {
    if (value === null || value === undefined) {
      return <span className="text-muted-foreground">N/A</span>;
    }
    if (typeof value === "boolean") {
      return <span className={value ? "text-green-500" : "text-red-400"}>{value ? "Yes" : "No"}</span>;
    }
    if (typeof value === "number") {
      return <span className="text-blue-400">{value}</span>;
    }
    if (typeof value === "string") {
      if (value.length > 50) {
        return <span className="text-foreground break-all">{value}</span>;
      }
      return <span className="text-foreground">{value}</span>;
    }
    if (Array.isArray(value)) {
      if (value.length === 0) {
        return <span className="text-muted-foreground">[]</span>;
      }
      return (
        <span className="text-muted-foreground">
          [{value.length} item{value.length > 1 ? "s" : ""}]
        </span>
      );
    }
    if (typeof value === "object") {
      const obj = value as Record<string, unknown>;
      const keys = Object.keys(obj);
      if (keys.length === 0) {
        return <span className="text-muted-foreground">{"{}"}</span>;
      }
      return <span className="text-muted-foreground">{"{...}"}</span>;
    }
    return String(value);
  };

  const renderDetailItem = (key: string, value: unknown, depth: number = 0): React.ReactNode => {
    const isNestedObject = typeof value === "object" && value !== null && !Array.isArray(value);
    const isArray = Array.isArray(value);
    
    if (isNestedObject && depth < 2) {
      const obj = value as Record<string, unknown>;
      const entries = Object.entries(obj);
      
      return (
        <div key={key} className="space-y-1">
          <div className="font-medium text-muted-foreground flex items-center gap-1">
            <span>{key}</span>
            <span className="text-xs text-muted-foreground/60">({entries.length})</span>
          </div>
          <div className="pl-3 border-l-2 border-border/50 space-y-1">
            {entries.map(([k, v]) => renderDetailItem(k, v, depth + 1))}
          </div>
        </div>
      );
    }

    if (isArray && depth < 2) {
      const arr = value as unknown[];
      if (arr.length > 0 && typeof arr[0] === "object" && arr[0] !== null) {
        return (
          <div key={key} className="space-y-1">
            <div className="font-medium text-muted-foreground flex items-center gap-1">
              <span>{key}</span>
              <span className="text-xs text-muted-foreground/60">({arr.length})</span>
            </div>
            <div className="pl-3 border-l-2 border-border/50 space-y-1">
              {arr.slice(0, 10).map((item, idx) => {
                if (typeof item === "object" && item !== null) {
                  const objItem = item as Record<string, unknown>;
                  return (
                    <div key={idx} className="bg-muted/20 rounded px-2 py-1 space-y-0.5">
                      {Object.entries(objItem).slice(0, 5).map(([k, v]) => (
                        <div key={k} className="flex gap-2 text-xs">
                          <span className="text-muted-foreground">{k}:</span>
                          <span className="text-foreground truncate max-w-[200px]">
                            {renderDataValue(v, depth + 1)}
                          </span>
                        </div>
                      ))}
                      {Object.keys(objItem).length > 5 && (
                        <div className="text-xs text-muted-foreground">
                          +{Object.keys(objItem).length - 5} more...
                        </div>
                      )}
                    </div>
                  );
                }
                return (
                  <div key={idx} className="text-xs text-foreground">
                    {renderDataValue(item, depth + 1)}
                  </div>
                );
              })}
              {arr.length > 10 && (
                <div className="text-xs text-muted-foreground">
                  +{arr.length - 10} more items...
                </div>
              )}
            </div>
          </div>
        );
      }
    }

    return (
      <div key={key} className="flex gap-2 text-xs">
        <span className="text-muted-foreground font-medium min-w-[80px]">{key}:</span>
        <span className="text-foreground font-mono break-all flex-1">
          {renderDataValue(value, depth)}
        </span>
      </div>
    );
  };

  return (
    <div
      className={`rounded-lg border ${
        step.status === "checking"
          ? "border-primary"
          : step.valid
            ? "border-green-500"
            : "border-red-500"
      }`}
    >
      <div
        className={`flex items-center gap-3 p-3 ${hasDetails ? "cursor-pointer hover:bg-muted/50" : ""}`}
        onClick={() => hasDetails && toggleStep(step.step)}
      >
        <div className="flex-shrink-0">{getStatusIcon()}</div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium truncate">{label}</div>
          {step.status === "done" && step.error && !isExpanded && (
            <div className={`text-xs mt-0.5 break-all ${step.valid ? "text-yellow-600" : "text-red-500"}`}>
              {step.error}
            </div>
          )}
        </div>
        {hasDetails && step.status === "done" && (
          <div className="flex-shrink-0">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        )}
      </div>

      {isExpanded && hasDetails && (
        <div className="px-3 pb-3 pt-0">
          <div className="bg-muted/30 rounded-md p-3 text-xs space-y-3">
            {step.error && (
              <div className="space-y-1">
                <div className="font-semibold text-red-500 uppercase tracking-wide text-[10px]">Error</div>
                <div className="text-red-400 font-mono break-all bg-red-500/10 rounded px-2 py-1">{step.error}</div>
              </div>
            )}

            {step.warnings && step.warnings.length > 0 && (
              <div className="space-y-1">
                <div className="font-semibold text-yellow-600 uppercase tracking-wide text-[10px]">Warnings</div>
                <div className="bg-yellow-500/10 rounded px-2 py-1">
                  <ul className="list-disc list-inside text-yellow-600/80">
                    {step.warnings.map((w, i) => (
                      <li key={i} className="break-all">{w}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {step.data && Object.keys(step.data).length > 0 && (
              <div className="space-y-2">
                <div className="font-semibold text-foreground uppercase tracking-wide text-[10px]">Details</div>
                <div className="space-y-2">
                  {Object.entries(step.data).map(([key, value]) => renderDetailItem(key, value))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function WelcomeScreen({ onComplete }: WelcomeScreenProps) {
  const [step, setStep] = useState<0 | 1 | 2 | 3 | 4>(0);
  const [countdown, setCountdown] = useState(5);
  const [validationSteps, setValidationSteps] = useState<ValidationStep[]>([]);
  const [setupSteps, setSetupSteps] = useState<ValidationStep[]>([]);
  const [currentMessage, setCurrentMessage] = useState<string>("");
  const [setupCurrentMessage, setSetupCurrentMessage] = useState<string>("");
  const [validationAllPassed, setValidationAllPassed] = useState(false);
  const [setupAllPassed, setSetupAllPassed] = useState(false);
  const [showDisagreeDialog, setShowDisagreeDialog] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [agreementText, setAgreementText] = useState<string>("");
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [validationStarted, setValidationStarted] = useState(false);
  const [setupStarted, setSetupStarted] = useState(false);
  const [expandedValidationSteps, setExpandedValidationSteps] = useState<Set<string>>(new Set());
  const [expandedSetupSteps, setExpandedSetupSteps] = useState<Set<string>>(new Set());
  const [showSkipDialog, setShowSkipDialog] = useState(false);
  const [showValidationContent, setShowValidationContent] = useState(false);
  const [showSetupContent, setShowSetupContent] = useState(false);
  const [hasScrolledToBottom, setHasScrolledToBottom] = useState(false);

  const wsRef = useRef<PiscesL1WelcomeWS | null>(null);
  const stepRef = useRef<0 | 1 | 2 | 3 | 4>(0);
  const isMountedRef = useRef(true);
  const validationStartTimeRef = useRef<number>(0);
  const setupStartTimeRef = useRef<number>(0);
  const agreementScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    stepRef.current = step;
  }, [step]);

  useEffect(() => {
    isMountedRef.current = true;
    wsRef.current = new PiscesL1WelcomeWS();

    wsRef.current.onConnect(() => {
      if (!isMountedRef.current) return;
      setIsWsConnected(true);
    });

    wsRef.current.onDisconnect(() => {
      if (!isMountedRef.current) return;
      setIsWsConnected(false);
    });

    wsRef.current.on("agreement", (msg: WelcomeServerMessage) => {
      if (!isMountedRef.current) return;
      if (msg.type === "agreement") {
        setAgreementText(msg.content);
      }
    });

    wsRef.current.on("checking", (msg: WelcomeServerMessage) => {
      if (!isMountedRef.current) return;
      if (msg.type === "checking") {
        setCurrentMessage(msg.message || `Checking ${STEP_LABELS[msg.step] || msg.step}...`);
        setValidationSteps((prev) => {
          const existing = prev.find((s) => s.step === msg.step);
          if (existing) {
            return prev.map((s) =>
              s.step === msg.step ? { ...s, status: "checking", message: msg.message } : s
            );
          }
          return [
            ...prev,
            {
              step: msg.step,
              message: msg.message || "",
              status: "checking" as const,
              valid: false,
              error: null,
              data: null,
              warnings: [],
            },
          ];
        });
        setSetupSteps((prev) => {
          const existing = prev.find((s) => s.step === msg.step);
          if (existing) {
            return prev.map((s) =>
              s.step === msg.step ? { ...s, status: "checking", message: msg.message } : s
            );
          }
          return [
            ...prev,
            {
              step: msg.step,
              message: msg.message || "",
              status: "checking" as const,
              valid: false,
              error: null,
              data: null,
              warnings: [],
            },
          ];
        });
        setSetupCurrentMessage(msg.message || `Setting up ${SETUP_STEP_LABELS[msg.step] || msg.step}...`);
      }
    });

    wsRef.current.on("result", (msg: WelcomeServerMessage) => {
      if (!isMountedRef.current) return;
      if (msg.type === "result") {
        const newStep = {
          step: msg.step,
          message: "",
          status: "done" as const,
          valid: msg.valid,
          error: msg.error,
          data: msg.data || null,
          warnings: msg.warnings || [],
        };

        setValidationSteps((prev) =>
          prev.map((s) => (s.step === msg.step ? { ...s, ...newStep } : s))
        );
        setSetupSteps((prev) =>
          prev.map((s) => (s.step === msg.step ? { ...s, ...newStep } : s))
        );

        if (!msg.valid && msg.error) {
          setExpandedValidationSteps((prev) => new Set(prev).add(msg.step));
          setExpandedSetupSteps((prev) => new Set(prev).add(msg.step));
        }
      }
    });

    wsRef.current.on("done", (msg: WelcomeServerMessage) => {
      if (!isMountedRef.current) return;
      if (msg.type === "done") {
        const currentStep = stepRef.current;
        const MIN_DISPLAY_TIME = 1500;

        if (currentStep === 2) {
          const elapsed = Date.now() - validationStartTimeRef.current;
          const remainingTime = Math.max(0, MIN_DISPLAY_TIME - elapsed);

          setTimeout(() => {
            if (!isMountedRef.current) return;
            setValidationAllPassed(msg.valid);
            if (msg.valid) {
              setExpandedValidationSteps(new Set());
            }
            setShowValidationContent(true);
          }, remainingTime);
        } else if (currentStep === 3) {
          const elapsed = Date.now() - setupStartTimeRef.current;
          const remainingTime = Math.max(0, MIN_DISPLAY_TIME - elapsed);

          setTimeout(() => {
            if (!isMountedRef.current) return;
            setSetupAllPassed(msg.valid);
            if (msg.valid) {
              setExpandedSetupSteps(new Set());
            }
            setShowSetupContent(true);
          }, remainingTime);
        }
      }
    });

    wsRef.current.on("first_launch_completed", (msg: WelcomeServerMessage) => {
      if (!isMountedRef.current) return;
      if (msg.type === "first_launch_completed" && msg.success) {
        onComplete();
      }
    });

    wsRef.current.on("error", (msg: WelcomeServerMessage) => {
      if (!isMountedRef.current) return;
      if (msg.type === "error") {
        const currentStep = stepRef.current;
        if (currentStep === 2) {
          setValidationError(msg.message);
        } else if (currentStep === 3) {
          setSetupError(msg.message);
        }
      }
    });

    wsRef.current.connect().catch(console.error);

    return () => {
      isMountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.disconnect();
        wsRef.current = null;
      }
    };
  }, [onComplete]);

  useEffect(() => {
    if (step === 1 && countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(countdown - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [step, countdown]);

  useEffect(() => {
    if (step === 1 && isWsConnected && !agreementText && wsRef.current && isMountedRef.current) {
      wsRef.current.getAgreement();
    }
  }, [step, isWsConnected, agreementText]);

  useEffect(() => {
    if (step === 2 && isWsConnected && !validationStarted && wsRef.current && isMountedRef.current) {
      setValidationStarted(true);
      setValidationSteps([]);
      setValidationAllPassed(false);
      setValidationError(null);
      setExpandedValidationSteps(new Set());
      setShowValidationContent(false);
      validationStartTimeRef.current = Date.now();
      wsRef.current.validateConfig();
    }
  }, [step, isWsConnected, validationStarted]);

  useEffect(() => {
    if (step === 3 && isWsConnected && !setupStarted && wsRef.current && isMountedRef.current) {
      setSetupStarted(true);
      setSetupSteps([]);
      setSetupAllPassed(false);
      setSetupError(null);
      setExpandedSetupSteps(new Set());
      setShowSetupContent(false);
      setupStartTimeRef.current = Date.now();
      wsRef.current.setupEnvironment();
    }
  }, [step, isWsConnected, setupStarted]);

  const handleStart = () => {
    setStep(1);
  };

  const handleAgree = () => {
    if (countdown === 0) {
      setStep(2);
    }
  };

  const handleValidationNext = () => {
    setStep(3);
  };

  const handleSetupNext = () => {
    setStep(4);
  };

  const handleValidationRetry = () => {
    setValidationStarted(false);
    setValidationSteps([]);
    setValidationAllPassed(false);
    setValidationError(null);
    setCurrentMessage("");
    setExpandedValidationSteps(new Set());
  };

  const handleSetupRetry = () => {
    setSetupStarted(false);
    setSetupSteps([]);
    setSetupAllPassed(false);
    setSetupError(null);
    setSetupCurrentMessage("");
    setExpandedSetupSteps(new Set());
  };

  const handleSetupSkip = () => {
    setShowSkipDialog(true);
  };

  const handleSkipConfirm = () => {
    setShowSkipDialog(false);
    setStep(4);
  };

  const handleSkipCancel = () => {
    setShowSkipDialog(false);
  };

  const handleDisagree = () => {
    setShowDisagreeDialog(true);
  };

  const handleDisagreeConfirm = () => {
    setShowDisagreeDialog(false);
    setStep(0);
    setCountdown(5);
    setHasScrolledToBottom(false);
  };

  const handleDisagreeCancel = () => {
    setShowDisagreeDialog(false);
  };

  const handleComplete = () => {
    if (wsRef.current && isWsConnected) {
      wsRef.current.completeFirstLaunch();
    } else {
      onComplete();
    }
  };

  const toggleValidationStep = (stepId: string) => {
    setExpandedValidationSteps((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(stepId)) {
        newSet.delete(stepId);
      } else {
        newSet.add(stepId);
      }
      return newSet;
    });
  };

  const toggleSetupStep = (stepId: string) => {
    setExpandedSetupSteps((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(stepId)) {
        newSet.delete(stepId);
      } else {
        newSet.add(stepId);
      }
      return newSet;
    });
  };

  const handleAgreementScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.target as HTMLDivElement;
    const { scrollTop, scrollHeight, clientHeight } = target;
    const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10;
    if (isAtBottom && !hasScrolledToBottom) {
      setHasScrolledToBottom(true);
    }
  };

  if (step === 0) {
    return (
      <GlobalOverlay layout="full">
        <div className="flex flex-col items-center justify-center h-full text-center">
          <h1 className="text-3xl font-bold mb-10 text-foreground">
            Welcome to Xi Studio
          </h1>

          <Button
            onClick={handleStart}
            size="lg"
            className="px-8"
          >
            Get Started
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </GlobalOverlay>
    );
  }

  if (step === 1) {
    return (
      <>
        <GlobalOverlay layout="full">
          <div className="flex flex-col h-full">
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-center">Service Agreement</h2>
            </div>
            <div className="flex-1 flex flex-col min-h-0">
              <div className="border border-border rounded-lg overflow-hidden mb-6 flex-1 min-h-0">
                <div
                  ref={agreementScrollRef}
                  onScroll={handleAgreementScroll}
                  className="p-4 h-full overflow-y-auto bg-muted/30 text-sm prose prose-sm max-w-none prose-headings:text-foreground prose-p:text-muted-foreground prose-h1:text-lg prose-h2:text-base prose-h3:text-sm prose-ul:list-disc prose-ol:list-decimal"
                >
                  {agreementText ? (
                    <ReactMarkdown>{agreementText}</ReactMarkdown>
                  ) : (
                    "Loading agreement..."
                  )}
                </div>
              </div>

              <div className="flex justify-center gap-4 flex-shrink-0 pt-2">
                <Button
                  onClick={handleDisagree}
                  size="lg"
                  className="px-8"
                >
                  Disagree
                </Button>
                <Button
                  onClick={handleAgree}
                  disabled={countdown > 0 || !agreementText || !hasScrolledToBottom}
                  size="lg"
                  className="px-8"
                >
                  {countdown > 0 ? `Agree (${countdown}s)` : !hasScrolledToBottom ? "Scroll to read" : "Agree"}
                </Button>
              </div>
            </div>
          </div>
        </GlobalOverlay>

        <Dialog open={showDisagreeDialog} onOpenChange={setShowDisagreeDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center justify-center gap-2">
                <AlertCircle className="h-5 w-5 text-red-500" />
                Decline Agreement?
              </DialogTitle>
              <DialogDescription>
                You must agree to the Service Agreement to use Xi Studio.
                Are you sure you want to decline?
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2">
              <Button onClick={handleDisagreeCancel}>
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

  const isValidationComplete = validationAllPassed && validationSteps.length > 0 && validationSteps.every(s => s.status === "done");
  const hasValidationFailed = validationSteps.some(s => s.status === "done" && !s.valid);

  if (step === 2) {
    return (
      <GlobalOverlay
        layout="split"
        leftContent={
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
          </div>
        }
        rightContent={
          <div className="flex flex-col h-full">
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-center">Validating Configuration</h2>
            </div>
            <div className="flex-1 flex flex-col min-h-0">
              {!showValidationContent ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 min-h-0">
                  <div className="relative w-24 h-10">
                    <Image
                      src="/load.svg"
                      alt="Loading"
                      fill
                      className="object-contain animate-pulse"
                      priority
                    />
                  </div>
                  <p className="text-sm text-muted-foreground text-center">
                    {validationSteps.length > 0 ? "Validating..." : "Starting validation..."}
                  </p>
                </div>
              ) : (
                <>
                  <div className="space-y-2 flex-1 overflow-y-auto min-h-0">
                    {validationSteps.filter(s => s.status !== "pending").map((s) => (
                      <ValidationStepItem
                        key={s.step}
                        step={s}
                        labels={STEP_LABELS}
                        expandedSteps={expandedValidationSteps}
                        toggleStep={toggleValidationStep}
                      />
                    ))}
                  </div>

                  <div className="mt-6 flex justify-center gap-4">
                    {(validationError || hasValidationFailed) && (
                      <Button
                        onClick={handleValidationRetry}
                        size="lg"
                        variant="outline"
                        className="px-8"
                      >
                        Retry
                      </Button>
                    )}
                    {isValidationComplete && (
                      <Button
                        onClick={handleValidationNext}
                        size="lg"
                        className="px-8"
                      >
                        Next
                        <ArrowRight className="ml-2 h-5 w-5" />
                      </Button>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        }
      />
    );
  }

  const isSetupComplete = setupAllPassed && setupSteps.length > 0 && setupSteps.every(s => s.status === "done");
  const hasSetupFailed = setupSteps.some(s => s.status === "done" && !s.valid);

  if (step === 3) {
    return (
      <>
      <GlobalOverlay
        layout="split"
        leftContent={
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
          </div>
        }
        rightContent={
          <div className="flex flex-col h-full">
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-center">Environment Setup</h2>
            </div>
            <div className="flex-1 flex flex-col min-h-0">
              {!showSetupContent ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 min-h-0">
                  <div className="relative w-24 h-10">
                    <Image
                      src="/load.svg"
                      alt="Loading"
                      fill
                      className="object-contain animate-pulse"
                      priority
                    />
                  </div>
                  <p className="text-sm text-muted-foreground text-center">
                    {setupSteps.length > 0 ? "Setting up..." : "Starting setup..."}
                  </p>
                </div>
              ) : (
                <>
                  <div className="space-y-2 flex-1 overflow-y-auto min-h-0">
                    {setupSteps.filter(s => s.status !== "pending").map((s) => (
                      <ValidationStepItem
                        key={s.step}
                        step={s}
                        labels={SETUP_STEP_LABELS}
                        expandedSteps={expandedSetupSteps}
                        toggleStep={toggleSetupStep}
                      />
                    ))}
                  </div>

                  <div className="mt-6 flex justify-center gap-4">
                    {(setupError || hasSetupFailed) && (
                      <>
                        <Button
                          onClick={handleSetupRetry}
                          size="lg"
                          variant="outline"
                          className="px-8"
                        >
                          Retry
                        </Button>
                        <Button
                          onClick={handleSetupSkip}
                          size="lg"
                          variant="outline"
                          className="px-8"
                        >
                          Skip
                        </Button>
                      </>
                    )}
                    {isSetupComplete && (
                      <Button
                        onClick={handleSetupNext}
                        size="lg"
                        className="px-8"
                      >
                        Next
                        <ArrowRight className="ml-2 h-5 w-5" />
                      </Button>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        }
      />

      <Dialog open={showSkipDialog} onOpenChange={setShowSkipDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center justify-center gap-2">
              <AlertCircle className="h-5 w-5 text-yellow-500" />
              Skip Environment Setup?
            </DialogTitle>
            <DialogDescription>
              Skipping may cause some features to not work. You can run setup manually from settings later.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button onClick={handleSkipCancel} variant="outline">
              Cancel
            </Button>
            <Button onClick={handleSkipConfirm} variant="destructive">
              Skip Anyway
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

if (step === 4) {
    return (
      <GlobalOverlay layout="full">
        <div className="flex flex-col items-center justify-center h-full text-center">
          <div className="flex justify-center mb-4">
            <CheckCircle2 className="h-16 w-16 text-green-500" />
          </div>

          <h1 className="text-2xl font-bold mb-4 text-foreground">
            Congratulations!
          </h1>

          <p className="text-base text-muted-foreground mb-6">
            You can now use Xi Studio
          </p>

          <Button
            onClick={handleComplete}
            size="lg"
            className="px-8"
          >
            Get Started
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </GlobalOverlay>
    );
  }

  return null;
}
