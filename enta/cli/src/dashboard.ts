/**
 * EnTA Dashboard — real-time terminal renderer with status, planning, and talk mode.
 *
 * Layout (top to bottom):
 *   1. Status — startup progress, model state
 *   2. Planning — main model's current plan
 *   3. Logs — recent events
 *   4. Talk transcript (when in talk mode)
 *   5. Input bar — always at the bottom, command hints only on "/"
 *
 * Rendering is debounced (50 ms).  Each ``render()`` writes the entire screen
 * as a single ``process.stdout.write()`` call.
 */

import chalk from "chalk";
import { EOL } from "node:os";

// ── Type definitions ─────────────────────────────────────────────

interface PhaseMessage {
  type: "phase";
  phase: string;
  iteration: number;
  total_iterations: number;
  dataset_path?: string;
  samples?: number;
}

interface RewardMessage {
  type: "reward";
  reward: number;
  capability_score?: number;
  iteration: number;
}

interface LogMessage {
  type: "log";
  level: string;
  message: string;
}

interface ErrorMessage {
  type: "error";
  message: string;
}

interface CompleteMessage {
  type: "complete";
  iterations: number;
  total_batches: number;
}

interface StatusMessage {
  type: "status";
  status: string;
  detail: string;
}

interface PlanMessage {
  type: "plan";
  plan: string;
  iteration?: number;
}

interface TalkMessage {
  type: "talk";
  from: string;
  text: string;
}

interface EngineMessage {
  type: "engine";
  action: "started" | "stopped" | "paused" | "resumed";
  detail?: string;
}

export type DashboardMessage =
  | PhaseMessage
  | RewardMessage
  | LogMessage
  | ErrorMessage
  | CompleteMessage
  | StatusMessage
  | PlanMessage
  | TalkMessage
  | EngineMessage;

// ── Dashboard ────────────────────────────────────────────────────

export class Dashboard {
  private iteration = 0;
  private totalIterations = 8;
  private phase = "init";
  private phaseDetail = "";
  private reward = 0;
  private previousReward = 0;
  private capabilityScore = 0;
  private logs: string[] = [];
  private maxLogLines = 6;
  private completed = false;
  private startTime = Date.now();
  private status = "initializing";
  private statusDetail = "Starting EnTA training pipeline...";
  private plan = "";
  private talkMessages: string[] = [];
  private maxTalkLines = 5;
  isTalkMode = false;
  talkTarget = "";
  private inputBuffer = "";

  private lastRender = "";
  private renderPending = false;
  private renderTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly DEBOUNCE_MS = 50;

  // ── Public API ────────────────────────────────────────────────

  handleMessage(msg: DashboardMessage): void {
    switch (msg.type) {
      case "phase":
        this.iteration = msg.iteration;
        this.totalIterations = msg.total_iterations;
        this.phase = msg.phase;
        this.phaseDetail = msg.dataset_path
          ? msg.dataset_path
          : msg.samples ? `${msg.samples} samples` : "";
        this.pushLog("info",
          `Iter ${this.iteration}/${this.totalIterations} → ${this.phase}${this.phaseDetail ? " | " + this.phaseDetail : ""}`);
        break;

      case "reward":
        this.previousReward = this.reward;
        this.reward = msg.reward;
        this.capabilityScore = msg.capability_score ?? this.capabilityScore;
        this.pushLog("info",
          `Reward: ${msg.reward.toFixed(4)}${msg.capability_score != null ? ` | capability: ${msg.capability_score.toFixed(4)}` : ""}`);
        break;

      case "log":    this.pushLog(msg.level, msg.message); break;
      case "error":  this.pushLog("error", msg.message); break;

      case "complete":
        this.completed = true;
        this.pushLog("info", `EnTA loop complete: ${msg.iterations} iterations, ${msg.total_batches} batches`);
        break;

      case "status":
        this.status = msg.status;
        this.statusDetail = msg.detail;
        break;

      case "plan":
        this.plan = msg.plan;
        this.status = "planning";
        break;

      case "talk":
        this.pushTalk(msg.from, msg.text);
        break;

      case "engine":
        this.pushLog("info", `Engine ${msg.action}${msg.detail ? ": " + msg.detail : ""}`);
        if (msg.action === "paused") this.status = "paused";
        else if (msg.action === "resumed") this.status = "training";
        else if (msg.action === "stopped") this.status = "idle";
        else if (msg.action === "started") this.status = "training";
        break;
    }
    this.scheduleRender();
  }

  setInputBuffer(buffer: string): void {
    this.inputBuffer = buffer;
    this.scheduleRender();
  }

  /** Render the final summary (plain text, no ANSI codes, doesn't clear screen). */
  renderFinal(exitCode: number): void {
    this.cancelRender();
    process.stdout.write("\x1B[?25h");
    process.stdout.write("\x1B[0m");
    const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
    const lines: string[] = [
      "",
      "━━━━━━━ EnTA Training Complete ──────────────────",
      exitCode === 0
        ? "✓ Status: Success"
        : `✗ Status: Failed (code ${exitCode})`,
      `  Iterations: ${this.iteration} / ${this.totalIterations}`,
      `  Final reward: ${this.reward.toFixed(4)}`,
      `  Capability score: ${this.capabilityScore.toFixed(4)}`,
      `  Elapsed: ${elapsed}s`,
      "─".repeat(40),
      ...this.logs.slice(-4),
      "",
    ];
    process.stdout.write("\n" + lines.join(EOL) + EOL);
  }

  restoreTerminal(): void {
    process.stdout.write("\x1B[?25h");
    process.stdout.write("\x1B[0m");
    process.stdout.write("\x1B[2J");
    process.stdout.write("\x1B[H");
  }

  // ── Internal helpers ──────────────────────────────────────────

  private pushLog(level: string, message: string): void {
    const prefix =
      level === "error" ? chalk.red("✗")
      : level === "warn"  ? chalk.yellow("⚠")
      :                      chalk.dim("●");
    const text = message.length > 120 ? message.slice(0, 117) + "..." : message;
    this.logs.push(`${prefix} ${text}`);
    if (this.logs.length > this.maxLogLines) {
      this.logs = this.logs.slice(-this.maxLogLines);
    }
  }

  private pushTalk(from: string, text: string): void {
    const label = from === "user" ? chalk.cyan("you")
      : from === "main" ? chalk.magenta("main")
      : chalk.yellow(from);
    this.talkMessages.push(`${label}: ${text}`);
    if (this.talkMessages.length > this.maxTalkLines) {
      this.talkMessages = this.talkMessages.slice(-this.maxTalkLines);
    }
  }

  // ── Rendering (debounced) ─────────────────────────────────────

  private scheduleRender(): void {
    if (this.renderPending) return;
    this.renderPending = true;
    this.renderTimer = setTimeout(() => {
      this.renderPending = false;
      this.renderTimer = null;
      this.doRender();
    }, this.DEBOUNCE_MS);
  }

  private cancelRender(): void {
    if (this.renderTimer !== null) {
      clearTimeout(this.renderTimer);
      this.renderTimer = null;
    }
    this.renderPending = false;
  }

  private doRender(): void {
    const output = this.buildOutput();
    if (output === this.lastRender) return;
    this.lastRender = output;
    process.stdout.write(output);
  }

  // ── Output construction ───────────────────────────────────────

  private getTerminalHeight(): number {
    return Math.max(16, Math.min(80, process.stdout.rows ?? 24));
  }

  private buildOutput(): string {
    const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
    const rewardDelta = this.reward - this.previousReward;
    const rewardStr =
      rewardDelta > 0  ? chalk.green(`${this.reward.toFixed(4)} ↑+${rewardDelta.toFixed(4)}`)
      : rewardDelta < 0 ? chalk.red(`${this.reward.toFixed(4)} ↓${rewardDelta.toFixed(4)}`)
      :                    chalk.white(this.reward.toFixed(4));

    // ── Status color / icon ──
    const statusColor = this.status === "model_ready" ? chalk.green
      : this.status === "training"  ? chalk.cyan
      : this.status === "planning"  ? chalk.magenta
      : this.status === "paused"    ? chalk.yellow
      : this.status === "talk"      ? chalk.magenta
      : this.status === "error"     ? chalk.red
      :                                chalk.dim;
    const statusIcon = this.status === "model_ready" ? "✓"
      : this.status === "training"  ? "▶"
      : this.status === "planning"  ? "◉"
      : this.status === "paused"    ? "⏸"
      : this.status === "talk"      ? "💬"
      : this.status === "error"     ? "✗"
      :                                "●";

    // ── 1. Status section (compact) ──
    const body: string[] = [
      chalk.bold("┌─ EnTA Training Dashboard ─────────────────────────────┐"),
      `  ${statusColor(statusIcon)} ${chalk.bold("Status:")} ${statusColor(this.status)}  ${chalk.dim(this.statusDetail)}`,
      `  ${chalk.cyan("Iter:")} ${this.iteration}/${this.totalIterations}    ${chalk.cyan("Phase:")} ${this.phase}${this.phaseDetail ? chalk.dim(" | " + this.phaseDetail) : ""}`,
      `  ${chalk.cyan("Reward:")} ${rewardStr}   ${chalk.cyan("Elapsed:")} ${elapsed}s`,
    ];
    if (this.capabilityScore > 0) {
      body.push(`  ${chalk.cyan("Capability:")} ${this.capabilityScore.toFixed(4)}`);
    }

    // ── 2. Planning section ──
    if (this.plan) {
      const planLines = this.plan.split("\n").slice(0, 3);
      body.push(chalk.dim("─── Plan ───"));
      for (const l of planLines) body.push(`  ${chalk.dim(l)}`);
    }

    // ── 3. Logs section ──
    if (this.logs.length > 0) {
      body.push(chalk.dim("─── Logs ───"));
      for (const l of this.logs) body.push(`  ${l}`);
    }

    // ── 4. Talk transcript ──
    if (this.isTalkMode && this.talkMessages.length > 0) {
      body.push(chalk.dim(`─── Talk: ${this.talkTarget} ───`));
      for (const m of this.talkMessages) body.push(`  ${m}`);
    }

    // ── 5. Input bar ──
    const modePrefix = this.isTalkMode
      ? chalk.magenta(`[talk:${this.talkTarget}]`)
      : chalk.cyan(">");

    // Command suggestions — only show when input starts with "/" (like Claude Code).
    const showSuggestions = !this.isTalkMode && this.inputBuffer.startsWith("/");
    const inputBar: string[] = [
      chalk.dim("─".repeat(58)),
      `  ${modePrefix} ${this.inputBuffer}${chalk.inverse(" ")}`,
    ];
    if (showSuggestions) {
      inputBar.push(`  ${chalk.dim("/pause  /resume  /stop  /talk <target>  /status  /quit")}`);
    } else if (this.isTalkMode) {
      inputBar.push(`  ${chalk.dim("type a message or /endtalk")}`);
    }
    inputBar.push(chalk.bold("└────────────────────────────────────────────────────────┘"));

    // ── Pad to fill terminal height ──
    const termHeight = this.getTerminalHeight();
    const paddingNeeded = Math.max(0, termHeight - body.length - inputBar.length);
    const padding = paddingNeeded > 0 ? Array(paddingNeeded).fill("") : [];

    const allLines = [...body, ...padding, ...inputBar];
    return "\x1B[H\x1B[2J" + allLines.join(EOL) + EOL;
  }
}