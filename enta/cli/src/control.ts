/**
 * EnTA Control Handler — raw-mode keyboard input with inline buffer editing.
 *
 * Supports two modes:
 * 1. **Command mode** (default) — input is parsed as commands (``/pause``, ``/talk teacher``, etc.)
 * 2. **Talk mode** — input is forwarded as a message to the active target (teacher / main model)
 *
 * The caller (``index.ts``) decides how to dispatch each mode.
 */

export type ControlCommand =
  | "pause" | "resume" | "stop" | "status" | "quit"
  | "talk" | "endtalk" | "engine";

export interface ParsedInput {
  /** Raw command: "pause", "talk", etc.  ``null`` when in talk mode. */
  command: ControlCommand | null;
  /** Optional argument after the command (e.g. ``"teacher1"`` for ``/talk teacher1``). */
  arg: string;
  /** When in talk mode, this is the message text to forward. */
  talkMessage: string;
  /** Whether the input is a message (talk mode) rather than a command. */
  isTalkMessage: boolean;
}

export type CommandCallback = (parsed: ParsedInput) => void;
type BufferChangeCallback = (buffer: string) => void;

// ── Terminal raw-mode constants ────────────────────────────────

const CTRL_C = "\x03";
const CTRL_D = "\x04";
const ENTER = "\r";
const BACKSPACE = "\x7f";
const ESC = "\x1b";

export class ControlHandler {
  private buffer = "";
  private callback: CommandCallback | null = null;
  private onBufferChange: BufferChangeCallback | null = null;
  private started = false;
  private rawData = "";
  private escapeSequenceTimer: ReturnType<typeof setTimeout> | null = null;

  /** ``true`` when in talk mode — all input is a message, not a command. */
  talkMode = false;
  /** Name of the talk target (e.g. ``"main"``, ``"teacher1"``). */
  talkTarget = "";

  // ── Public API ────────────────────────────────────────────────

  onCommand(cb: CommandCallback): void {
    this.callback = cb;
  }

  onInputChange(cb: BufferChangeCallback): void {
    this.onBufferChange = cb;
  }

  getBuffer(): string {
    return this.buffer;
  }

  setBuffer(text: string): void {
    this.buffer = text;
    this.onBufferChange?.(text);
  }

  /** Start raw-mode listener. */
  start(): void {
    if (this.started) return;
    this.started = true;

    const stdin = process.stdin;
    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding("utf8");

    stdin.on("data", (data: string) => {
      this.rawData += data;

      if (this.rawData.startsWith(ESC)) {
        if (this.escapeSequenceTimer) clearTimeout(this.escapeSequenceTimer);
        this.escapeSequenceTimer = setTimeout(() => {
          this.handleRaw(this.rawData);
          this.rawData = "";
        }, 10);
        return;
      }

      this.handleRaw(this.rawData);
      this.rawData = "";
    });

    stdin.on("error", () => {
      this.started = false;
    });
  }

  /** Stop raw-mode listener and restore terminal. */
  stop(): void {
    if (!this.started) return;
    this.started = false;
    try { process.stdin.setRawMode(false); } catch { /* ignore */ }
    process.stdin.pause();
  }

  /** Enter talk mode with a specific target. */
  enterTalkMode(target: string): void {
    this.talkMode = true;
    this.talkTarget = target;
    this.buffer = "";
    this.onBufferChange?.("");
  }

  /** Exit talk mode and return to command mode. */
  exitTalkMode(): void {
    this.talkMode = false;
    this.talkTarget = "";
    this.buffer = "";
    this.onBufferChange?.("");
  }

  // ── Internals ────────────────────────────────────────────────

  private handleRaw(data: string): void {
    switch (data) {
      case ENTER:
        this.handleEnter();
        return;
      case BACKSPACE:
      case "\b":
        if (this.buffer.length > 0) {
          this.buffer = this.buffer.slice(0, -1);
          this.onBufferChange?.(this.buffer);
        }
        return;
      case CTRL_C:
      case CTRL_D:
        if (this.talkMode) {
          // Ctrl+C/D in talk mode — exit talk mode.
          this.exitTalkMode();
          this.callback?.({
            command: "endtalk",
            arg: "",
            talkMessage: "",
            isTalkMessage: false,
          });
        } else {
          this.callback?.({
            command: "quit",
            arg: "",
            talkMessage: "",
            isTalkMessage: false,
          });
        }
        return;
      default:
        if (data.startsWith(ESC)) return; // ignore escape sequences
        if (data >= " " || data > "\x7f") {
          this.buffer += data;
          this.onBufferChange?.(this.buffer);
        }
        return;
    }
  }

  private handleEnter(): void {
    const trimmed = this.buffer.trim();
    this.buffer = "";
    this.onBufferChange?.("");

    if (!trimmed) return;

    // ── Talk mode: everything is a message (except /endtalk) ──
    if (this.talkMode) {
      const isEnd = trimmed.toLowerCase() === "/endtalk" || trimmed.toLowerCase() === "endtalk";
      if (isEnd) {
        this.exitTalkMode();
        this.callback?.({
          command: "endtalk",
          arg: "",
          talkMessage: "",
          isTalkMessage: false,
        });
      } else {
        this.callback?.({
          command: null,
          arg: "",
          talkMessage: trimmed,
          isTalkMessage: true,
        });
      }
      return;
    }

    // ── Command mode: parse as command ──
    const cmd = trimmed.startsWith("/") ? trimmed.slice(1) : trimmed;
    const parts = cmd.split(/\s+/);
    const verb = parts[0]?.toLowerCase() ?? "";
    const arg = parts.slice(1).join(" ");

    const parsed = this.parseCommand(verb);
    if (parsed || verb === "talk") {
      // Special handling for /talk — it's always valid.
      if (verb === "talk") {
        this.callback?.({
          command: "talk",
          arg: arg || "main",
          talkMessage: "",
          isTalkMessage: false,
        });
      } else {
        this.callback?.({
          command: parsed,
          arg,
          talkMessage: "",
          isTalkMessage: false,
        });
      }
    }
    // Unknown commands are silently ignored (no spam).
  }

  private parseCommand(verb: string): ControlCommand | null {
    switch (verb) {
      case "pause":   case "p": return "pause";
      case "resume":  case "r": return "resume";
      case "stop":    case "s": return "stop";
      case "status":            return "status";
      case "quit":    case "q": return "quit";
      case "endtalk":           return "endtalk";
      case "engine":            return "engine";
      case "talk":              return "talk";
      default:                  return null;
    }
  }
}