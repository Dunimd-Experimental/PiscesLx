#!/usr/bin/env node

/**
 * EnTA CLI — Real-time training dashboard for the EnTA outer loop.
 *
 * Forwards all arguments to ``python manage.py train <args> --jsonlines``.
 * Supports raw-mode keyboard input with command mode and talk mode.
 *
 * Key design principle: **never exit on child process errors**.  If the
 * Python training process crashes, the CLI stays alive, shows the error
 * in the dashboard, and waits for the user to explicitly ``/quit``.
 */

import { spawn, ChildProcess } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { Dashboard } from "./dashboard.js";
import { ControlHandler, type ParsedInput } from "./control.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..", "..", "..");
const PYTHON = process.platform === "win32" ? "python" : "python3";

const forwardArgs = process.argv.slice(2);
const pythonArgs = [resolve(ROOT, "manage.py"), "train", ...forwardArgs];
if (!forwardArgs.some((a) => a === "--jsonlines")) {
  pythonArgs.push("--jsonlines");
}

// ── Bootstrap ─────────────────────────────────────────────────

const dashboard = new Dashboard();
const control = new ControlHandler();
let child: ChildProcess | null = null;
let childExited = false;
let cleanedUp = false;

// ── Safe cleanup (only called on explicit user quit) ─────────

function cleanup(exitCode?: number): void {
  if (cleanedUp) return;
  cleanedUp = true;
  try { control.stop(); } catch { /* ignore */ }
  try {
    process.stdout.write("\x1B[?25h");
    process.stdout.write("\x1B[0m");
  } catch { /* ignore */ }
  if (child && !child.killed) {
    try { child.kill("SIGTERM"); } catch { /* ignore */ }
  }
  if (exitCode !== undefined) {
    process.exit(exitCode);
  }
}

// ── Input handling ───────────────────────────────────────────

function sendToPython(obj: unknown): void {
  if (child?.stdin && !child.killed && !childExited) {
    try {
      child.stdin.write(JSON.stringify(obj) + "\n");
    } catch { /* ignore */ }
  }
}

control.onInputChange((buffer) => {
  dashboard.setInputBuffer(buffer);
});

control.onCommand((parsed: ParsedInput) => {
  if (parsed.isTalkMessage) {
    sendToPython({ type: "talk", from: "user", text: parsed.talkMessage });
    dashboard.handleMessage({
      type: "talk", from: "user", text: parsed.talkMessage,
    });
    return;
  }

  switch (parsed.command) {
    case "quit":
      cleanup(0);
      break;

    case "talk":
      control.enterTalkMode(parsed.arg || "main");
      dashboard.isTalkMode = true;
      dashboard.talkTarget = parsed.arg || "main";
      dashboard.handleMessage({ type: "log", level: "info",
        message: `Entering talk mode with '${parsed.arg || "main"}'. Type /endtalk to exit.` });
      if (!childExited) {
        sendToPython({ type: "command", command: "talk", target: parsed.arg || "main" });
      }
      break;

    case "endtalk":
      control.exitTalkMode();
      dashboard.isTalkMode = false;
      dashboard.talkTarget = "";
      dashboard.handleMessage({ type: "log", level: "info", message: "Exited talk mode." });
      if (!childExited) {
        sendToPython({ type: "command", command: "endtalk" });
      }
      break;

    case "pause":
    case "resume":
    case "stop":
    case "status":
    case "engine":
      if (!childExited) {
        sendToPython({ type: "command", command: parsed.command, arg: parsed.arg });
      }
      break;
  }
});

// ── Start Python training process ────────────────────────────

try {
  child = spawn(PYTHON, pythonArgs, {
    cwd: ROOT,
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
} catch (err) {
  dashboard.handleMessage({ type: "error", message: `Failed to spawn Python: ${err}` });
  dashboard.handleMessage({ type: "log", level: "info",
    message: "Type /quit to exit the CLI." });
  childExited = true;
}

// ── Handle stdout (JSON lines) ────────────────────────────────

if (child?.stdout) {
  child.stdout.setEncoding("utf-8");
  let buffer = "";
  child.stdout.on("data", (chunk: string) => {
    buffer += chunk;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const msg = JSON.parse(trimmed);
        (dashboard as any).handleMessage(msg);
      } catch {
        dashboard.handleMessage({ type: "log", level: "raw", message: trimmed });
      }
    }
  });
}

// ── Handle stderr ─────────────────────────────────────────────

if (child?.stderr) {
  child.stderr.setEncoding("utf-8");
  child.stderr.on("data", (chunk: string) => {
    const lines = chunk.split("\n");
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) {
        dashboard.handleMessage({ type: "log", level: "raw", message: trimmed });
      }
    }
  });
}

// ── Handle child exit (keep CLI alive!) ───────────────────────

if (child) {
  child.on("exit", (code, signal) => {
    childExited = true;
    const exitCode = code ?? (signal ? 1 : -1);
    dashboard.handleMessage({ type: "status", status: "error",
      detail: `Python process exited (code ${exitCode})` });
    dashboard.handleMessage({ type: "log", level: "info",
      message: "Training process stopped. Type /quit to exit the CLI." });
  });
  child.on("error", (err) => {
    childExited = true;
    dashboard.handleMessage({ type: "error", message: `Python process error: ${err.message}` });
    dashboard.handleMessage({ type: "log", level: "info",
      message: "Type /quit to exit the CLI." });
  });
}

// ── Start raw-mode input listener ─────────────────────────────

control.start();

// ── Process-level signals (only these trigger actual exit) ────

process.on("SIGINT", () => {
  dashboard.handleMessage({ type: "log", level: "info", message: "SIGINT received, shutting down..." });
  cleanup(130);
});
process.on("SIGTERM", () => cleanup(143));
process.on("uncaughtException", (err) => {
  dashboard.handleMessage({ type: "error", message: `Uncaught exception: ${err.message}` });
  cleanup(1);
});
process.on("unhandledRejection", (err) => {
  dashboard.handleMessage({ type: "error", message: `Unhandled rejection: ${err}` });
  cleanup(1);
});