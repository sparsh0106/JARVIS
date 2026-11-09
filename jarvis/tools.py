"""Capability Tool Registry, Dynamic Command Policy & Executor, Filesystem, Desktop, Memory, and AST Calculator."""

from __future__ import annotations

import ast
import fnmatch
import json
import math
import operator
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .memory import FAISSMemoryStore, VALID_MEMORY_TYPES

# ---------------------------------------------------------------------------
# Safe Deterministic AST Calculator
# ---------------------------------------------------------------------------

_SAFE_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
_SAFE_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_SAFE_MATH_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
    "round": round, "ceil": math.ceil, "floor": math.floor,
}
_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT, MAX_ABS_OPERAND = 1000.0, 1e100


def _eval_ast_node(node: ast.AST, depth: int = 0) -> float | int:
    if depth > 32:
        raise ValueError("Expression is too deeply nested.")
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown identifier: {node.id}")
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("Unsupported unary operator.")
        return op_fn(_eval_ast_node(node.operand, depth + 1))
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, depth + 1)
        right = _eval_ast_node(node.right, depth + 1)
        if abs(float(left)) > MAX_ABS_OPERAND or abs(float(right)) > MAX_ABS_OPERAND:
            raise ValueError("Operand magnitude exceeds safe limit.")
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > MAX_EXPONENT:
                raise ValueError(f"Exponent {right} exceeds maximum safe limit ({int(MAX_EXPONENT)}).")
            if abs(float(left)) > 10.0 and float(right) > 300.0:
                raise ValueError("Power result exceeds safe numeric range.")
            return operator.pow(left, right)
        op_fn = _SAFE_BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("Unsupported binary operator.")
        return op_fn(left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_MATH_FUNCS:
            raise ValueError("Only basic math functions (sqrt, sin, cos, log, etc.) are allowed.")
        return _SAFE_MATH_FUNCS[node.func.id](*[_eval_ast_node(a, depth + 1) for a in node.args])
    raise ValueError("Unsupported syntax in arithmetic expression.")


def calculate(expression: str) -> str:
    """Safely evaluate arithmetic expressions via AST parsing with exponent caps."""
    expr = re.sub(r"^(?:calculate|compute|evaluate|what\s+is)\s+", "", str(expression).strip(), flags=re.I).strip()
    if not expr:
        return "Empty expression."
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_ast_node(tree)
        val = float(result)
        if not math.isfinite(val):
            return "The result is not finite."
        return str(int(round(val))) if (isinstance(result, int) or val.is_integer()) else str(round(val, 10)).rstrip("0").rstrip(".")
    except ZeroDivisionError:
        return "Division by zero is undefined."
    except ValueError as exc:
        return f"Calculation error: {exc}"
    except Exception:
        return "Only safe arithmetic expressions are allowed."


# ---------------------------------------------------------------------------
# Command Security Policy & Data Models
# ---------------------------------------------------------------------------

_SHELL_CHAIN_PATTERN = re.compile(r"(?:;|&&|\|\||\|`|\$\(|`|\n|\r|\x00)")
_SHELL_OPERATORS = {";", "|", "&&", "||", ">", ">>", "<", "<<", "&"}

_DENIED_EXECUTABLES = {
    "sudo", "su", "doas", "pkexec", "visudo", "passwd", "chown", "chmod", "chgrp",
    "mkfs", "fdisk", "parted", "dd", "mount", "umount", "shutdown", "reboot",
    "poweroff", "halt", "init", "telinit", "iptables", "nft", "ufw",
    "nc", "netcat", "ncat", "socat", "curl", "wget", "ssh", "scp", "sftp", "rsync",
    "eval", "exec", "sh", "bash", "zsh", "dash", "fish", "csh", "tcsh", "ksh",
    "cmd", "powershell", "pwsh",
}

_SIMPLE_READONLY_EXECUTABLES = {
    "lscpu", "lsmem", "lsblk", "lspci", "lsusb", "free", "uptime", "uname",
    "hostname", "whoami", "id", "arch", "nproc", "sensors", "acpi", "locale",
    "ps", "pgrep", "pidof", "df", "ss", "pwd", "which", "whereis",
    "systeminfo", "tasklist",
}

_WORKSPACE_READONLY_FILE_COMMANDS = {
    "ls", "dir", "stat", "file", "du", "wc", "cat", "head", "tail", "grep", "rg", "find",
}

_NVIDIA_SMI_MUTATING_FLAGS = {
    "-r", "--gpu-reset", "-pm", "--persistence-mode", "-pl", "--power-limit",
    "-ac", "--applications-clocks", "-lgc", "--lock-gpu-clocks",
    "-rgc", "--reset-gpu-clocks", "-c", "--compute-mode", "-e", "--ecc-config",
}

_GIT_READONLY_SUBCOMMANDS = {"status", "log", "diff", "show", "rev-parse", "describe", "ls-files", "shortlog", "blame", "reflog"}
_GIT_REVERSIBLE_SUBCOMMANDS = {"add", "commit", "stash", "fetch", "pull", "switch"}
_GIT_DESTRUCTIVE_SUBCOMMANDS = {"reset", "clean", "rm", "push", "rebase", "merge"}

_ROS2_READONLY_VERBS = {
    "node": {"list", "info"}, "topic": {"list", "info", "type", "find", "bw", "hz", "delay", "echo"},
    "service": {"list", "type", "info", "find"}, "param": {"list", "get", "describe", "dump"},
    "interface": {"list", "show", "package", "packages", "proto"}, "pkg": {"list", "prefix", "executables", "xml"},
    "action": {"list", "info"}, "lifecycle": {"nodes", "get", "list"}, "bag": {"info"},
}
_ROS2_DESTRUCTIVE_VERBS = {
    "topic": {"pub"}, "service": {"call"}, "param": {"set", "delete", "load"},
    "action": {"send_goal"}, "lifecycle": {"set"},
}

_SAFE_ENV_KEYS = {
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
    "DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
    "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_PREFIX",
    "ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION", "ROS_DOMAIN_ID",
    "ROS_LOCALHOST_ONLY", "AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "COLCON_PREFIX_PATH",
    "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
}

APP_COMMANDS: Dict[str, Dict[str, List[str]]] = {
    "linux": {
        "terminal": ["x-terminal-emulator"], "calculator": ["gnome-calculator"],
        "vscode": ["code"], "vs code": ["code"], "chrome": ["google-chrome"],
        "firefox": ["firefox"], "sublime": ["subl"], "spotify": ["spotify"],
        "vlc": ["vlc"], "notepad": ["gedit"], "gedit": ["gedit"], "files": ["xdg-open", "."],
    },
    "darwin": {
        "terminal": ["open", "-a", "Terminal"], "calculator": ["open", "-a", "Calculator"],
        "vscode": ["open", "-a", "Visual Studio Code"], "vs code": ["open", "-a", "Visual Studio Code"],
        "chrome": ["open", "-a", "Google Chrome"], "firefox": ["open", "-a", "Firefox"], "files": ["open", "."],
    },
    "windows": {
        "calculator": ["calc.exe"], "notepad": ["notepad.exe"], "vscode": ["code"],
        "vs code": ["code"], "files": ["explorer.exe", "."],
    },
}


@dataclass
class CommandRequest:
    """Structured command execution request generated by the model."""
    executable: str
    args: List[str] = field(default_factory=list)
    cwd: Optional[str] = None
    purpose: Optional[str] = None
    timeout_seconds: int = 10
    confirm: bool = False


@dataclass
class PolicyDecision:
    """Deterministic validation result from CommandPolicyValidator."""
    allowed: bool
    permission_level: str  # "read_only" | "reversible" | "destructive" | "denied"
    requires_confirmation: bool = False
    reason: str = ""
    executable: str = ""
    args: List[str] = field(default_factory=list)
    cwd: Optional[Path] = None


@dataclass
class CommandResult:
    """Structured execution result returned to JarvisAgent."""
    ok: bool
    status: str
    command: List[str]
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    truncated: bool = False
    permission_level: str = "read_only"
    requires_confirmation: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ok": self.ok, "status": self.status, "command": self.command, "exit_code": self.exit_code,
            "stdout": self.stdout, "stderr": self.stderr, "duration_ms": round(self.duration_ms, 2),
            "timed_out": self.timed_out, "truncated": self.truncated, "permission_level": self.permission_level,
            "requires_confirmation": self.requires_confirmation,
        }
        if self.error:
            d["error"] = self.error
        return d


def _is_path_within_roots(candidate: Path, roots: Sequence[Path]) -> bool:
    resolved = candidate.resolve()
    for root in roots:
        root_resolved = root.resolve()
        try:
            if resolved.is_relative_to(root_resolved):
                return True
        except AttributeError:
            if str(resolved) == str(root_resolved) or str(resolved).startswith(str(root_resolved) + os.sep):
                return True
    return False


class CommandPolicyValidator:
    """Validates executables, subcommands, flags, and workspace path boundaries before execution."""

    def __init__(self, workspace_roots: Sequence[Path], require_confirmation_for_destructive: bool = True):
        self.workspace_roots: Tuple[Path, ...] = tuple(r.resolve() for r in workspace_roots)
        self.primary_root: Path = self.workspace_roots[0]
        self.require_confirmation = require_confirmation_for_destructive

    def resolve_safe_path(self, path_str: str, base_dir: Optional[Path] = None) -> Path:
        base = (base_dir or self.primary_root).resolve()
        raw = Path(path_str)
        candidate = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
        if not _is_path_within_roots(candidate, self.workspace_roots):
            raise PermissionError(f"Path '{path_str}' resolves outside approved workspace roots.")
        return candidate

    def _validate_path_args(self, args: List[str], cwd: Path, allow_missing: bool = False) -> Optional[str]:
        for arg in args:
            if not arg or arg.startswith("-"):
                continue
            try:
                self.resolve_safe_path(arg, base_dir=cwd)
                if not allow_missing and ".." in Path(arg).parts:
                    return f"Path traversal '{arg}' is not permitted."
            except PermissionError as exc:
                return str(exc)
        return None

    def _allow(self, exec_name: str, args: List[str], cwd: Path, level: str = "read_only") -> PolicyDecision:
        return PolicyDecision(allowed=True, permission_level=level, executable=exec_name, args=args, cwd=cwd)

    def _deny(self, exec_name: str, args: List[str], cwd: Path, reason: str) -> PolicyDecision:
        return PolicyDecision(allowed=False, permission_level="denied", reason=reason, executable=exec_name, args=args, cwd=cwd)

    def validate(self, request: CommandRequest) -> PolicyDecision:
        raw_exec = (request.executable or "").strip()
        if not raw_exec:
            return self._deny("", [], self.primary_root, "Executable name must be a non-empty string.")

        if _SHELL_CHAIN_PATTERN.search(raw_exec) or any(op in raw_exec for op in (";", "|", "&&", "||", ">", "<", "`", "$(")):
            return self._deny(raw_exec, [], self.primary_root, "Shell chaining, pipes, and redirection are not permitted in executable.")

        args = list(request.args or [])
        if (" " in raw_exec or "\t" in raw_exec) and not args:
            try:
                parts = shlex.split(raw_exec)
                if parts:
                    raw_exec, args = parts[0], parts[1:]
            except ValueError as exc:
                return self._deny(raw_exec, [], self.primary_root, f"Malformed command string: {exc}")

        if ".." in Path(raw_exec).parts or "/" in raw_exec or "\\" in raw_exec:
            return self._deny(raw_exec, args, self.primary_root, "Executable must be a command name, not a relative or arbitrary file path.")

        base_exec = Path(raw_exec).name.lower()
        if base_exec.endswith(".exe"):
            base_exec = base_exec[:-4]

        normalized_args: List[str] = []
        for idx, item in enumerate(args):
            if not isinstance(item, (str, int, float)) or isinstance(item, bool):
                return self._deny(base_exec, [], self.primary_root, f"Argument at index {idx} must be a string.")
            s_arg = str(item)
            if any(c in s_arg for c in ("\x00", "\n", "\r")):
                return self._deny(base_exec, [], self.primary_root, "Null bytes and control newlines are not permitted in arguments.")
            if s_arg.strip() in _SHELL_OPERATORS or _SHELL_CHAIN_PATTERN.search(s_arg):
                return self._deny(base_exec, [], self.primary_root, f"Shell operator or command injection pattern rejected: {s_arg!r}")
            normalized_args.append(s_arg)

        try:
            resolved_cwd = self.resolve_safe_path(request.cwd) if request.cwd else self.primary_root
        except PermissionError as exc:
            return self._deny(base_exec, normalized_args, self.primary_root, str(exc))

        if not resolved_cwd.exists() or not resolved_cwd.is_dir():
            return self._deny(base_exec, normalized_args, resolved_cwd, f"Working directory '{request.cwd}' does not exist.")

        if base_exec in _DENIED_EXECUTABLES:
            return self._deny(base_exec, normalized_args, resolved_cwd, f"Executable '{base_exec}' is denied by security policy.")

        decision = self._evaluate_command_policy(base_exec, normalized_args, resolved_cwd)
        if decision.allowed and decision.permission_level == "destructive" and self.require_confirmation and not request.confirm:
            decision.requires_confirmation = True

        return decision

    def _evaluate_command_policy(self, base_exec: str, args: List[str], cwd: Path) -> PolicyDecision:
        if base_exec in _SIMPLE_READONLY_EXECUTABLES:
            return self._allow(base_exec, args, cwd)

        if base_exec == "nvidia-smi":
            for arg in args:
                if arg.split("=")[0].lower() in _NVIDIA_SMI_MUTATING_FLAGS:
                    return self._deny(base_exec, args, cwd, f"Mutating nvidia-smi flag '{arg}' is not permitted.")
            return self._allow(base_exec, args, cwd)

        if base_exec == "top":
            safe = list(args)
            safe = (["-b", "-n", "1", *safe] if "-b" not in safe else ([*safe, "-n", "1"] if "-n" not in safe else safe))
            return self._allow(base_exec, safe, cwd)

        if base_exec == "date":
            return self._deny(base_exec, args, cwd, "Setting system clock via 'date' is forbidden.") if any(a in {"-s", "--set"} or a.startswith("--set=") for a in args) else self._allow(base_exec, args, cwd)

        if base_exec == "timedatectl":
            non_flags = [a for a in args if not a.startswith("-")]
            return self._allow(base_exec, args, cwd) if (not non_flags or non_flags[0] in {"status", "show", "list-timezones"}) else self._deny(base_exec, args, cwd, f"timedatectl subcommand '{non_flags[0]}' modifies system time.")

        if base_exec == "ip":
            return self._deny(base_exec, args, cwd, "Modifying network routes/interfaces is forbidden.") if any(v in [a.lower() for a in args] for v in {"add", "del", "delete", "set", "flush", "change", "replace"}) else self._allow(base_exec, args, cwd)

        if base_exec == "ping":
            safe = list(args)
            flag = "-n" if platform.system().lower() == "windows" else "-c"
            return self._allow(base_exec, [flag, "3", *safe] if flag not in safe else safe, cwd)

        if base_exec == "systemctl":
            subcmd = next((a for a in args if not a.startswith("-")), "status")
            if subcmd in {"status", "is-active", "is-enabled", "list-units", "list-unit-files", "list-timers", "show"}:
                return self._allow(base_exec, [*args, "--no-pager"] if "--no-pager" not in args else list(args), cwd)
            if subcmd in {"start", "stop", "restart", "reload", "enable", "disable"}:
                return self._allow(base_exec, args, cwd, "destructive")
            return self._deny(base_exec, args, cwd, f"Unsupported systemctl subcommand '{subcmd}'.")

        if base_exec == "journalctl":
            safe = list(args)
            if "--no-pager" not in safe: safe.append("--no-pager")
            if "-n" not in safe and "--lines" not in safe: safe.extend(["-n", "50"])
            return self._allow(base_exec, safe, cwd)

        if base_exec in _WORKSPACE_READONLY_FILE_COMMANDS:
            if base_exec == "tail" and any(f in args for f in ("-f", "--follow", "-F")):
                return self._deny(base_exec, args, cwd, "'tail -f' follow mode is not permitted.")
            if base_exec == "find":
                if any(f in args for f in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls")):
                    return self._deny(base_exec, args, cwd, "Mutating find flags are not permitted.")
                path_args = [a for a in args if not a.startswith("-") and a not in {"(", ")", "!", ","}]
                path_err = self._validate_path_args(path_args, cwd)
            elif base_exec in {"grep", "rg"}:
                non_flags = [a for a in args if not a.startswith("-")]
                path_err = self._validate_path_args(non_flags[1:], cwd)
            else:
                path_err = self._validate_path_args(args, cwd)
            return self._deny(base_exec, args, cwd, path_err) if path_err else self._allow(base_exec, args, cwd)

        if base_exec == "git":
            return self._evaluate_git_policy(base_exec, args, cwd)

        if base_exec in {"python", "python3", "pytest"}:
            return self._evaluate_python_policy(base_exec, args, cwd)

        if base_exec == "ros2":
            return self._evaluate_ros2_policy(base_exec, args, cwd)

        if base_exec in {"mkdir", "touch", "cp", "mv"}:
            path_err = self._validate_path_args(args, cwd, allow_missing=True)
            return self._deny(base_exec, args, cwd, path_err) if path_err else self._allow(base_exec, args, cwd, "reversible")

        if base_exec in {"rm", "rmdir"}:
            non_flags = [a for a in args if not a.startswith("-")]
            if not non_flags:
                return self._deny(base_exec, args, cwd, f"'{base_exec}' requires at least one target path.")
            path_err = self._validate_path_args(non_flags, cwd, allow_missing=True)
            if path_err:
                return self._deny(base_exec, args, cwd, path_err)
            for t in non_flags:
                if self.resolve_safe_path(t, base_dir=cwd) in self.workspace_roots:
                    return self._deny(base_exec, args, cwd, "Deleting a workspace root directory is forbidden.")
            return self._allow(base_exec, args, cwd, "destructive")

        if base_exec in {"kill", "pkill", "killall", "pip", "pip3", "apt", "apt-get", "dnf", "pacman"}:
            if base_exec in {"pip", "pip3"}:
                sub = next((a for a in args if not a.startswith("-")), "")
                if sub in {"list", "show", "check", "freeze"} or "--version" in args:
                    return self._allow(base_exec, args, cwd)
            return self._allow(base_exec, args, cwd, "destructive")

        return self._deny(base_exec, args, cwd, f"Executable '{base_exec}' is not permitted by the command policy.")

    def _evaluate_git_policy(self, base_exec: str, args: List[str], cwd: Path) -> PolicyDecision:
        if not args or args[0] in {"--version", "-v", "--help", "-h"}:
            return self._allow(base_exec, args, cwd)
        if "-c" in args or "--exec-path" in args:
            return self._deny(base_exec, args, cwd, "Git runtime config overrides (-c / --exec-path) are not permitted.")
        non_flags = [a for a in args if not a.startswith("-")]
        if not non_flags:
            return self._allow(base_exec, args, cwd)
        subcmd, rest = non_flags[0].lower(), args[args.index(non_flags[0]) + 1 :]
        if subcmd in _GIT_READONLY_SUBCOMMANDS or (subcmd == "config" and any(f in rest for f in ("--get", "--list", "-l"))):
            return self._allow(base_exec, args, cwd)
        if subcmd == "branch":
            return self._allow(base_exec, args, cwd, "destructive" if any(f in rest for f in ("-d", "-D", "--delete", "-m", "-M", "-c", "-C")) else ("read_only" if all(a.startswith("-") for a in rest) else "reversible"))
        if subcmd == "remote":
            return self._allow(base_exec, args, cwd, "read_only" if not rest or rest[0] in {"-v", "--verbose", "show"} else "destructive")
        if subcmd == "tag":
            return self._allow(base_exec, args, cwd, "read_only" if not rest or rest[0] in {"-l", "--list"} else "reversible")
        if subcmd == "checkout":
            return self._allow(base_exec, args, cwd, "destructive" if any(f in rest for f in ("-f", "--force")) else "reversible")
        if subcmd in _GIT_REVERSIBLE_SUBCOMMANDS:
            return self._allow(base_exec, args, cwd, "reversible")
        if subcmd in _GIT_DESTRUCTIVE_SUBCOMMANDS:
            return self._allow(base_exec, args, cwd, "destructive")
        return self._deny(base_exec, args, cwd, f"Git subcommand '{subcmd}' is not permitted by policy.")

    def _evaluate_python_policy(self, base_exec: str, args: List[str], cwd: Path) -> PolicyDecision:
        if base_exec == "pytest":
            path_err = self._validate_path_args(args, cwd)
            return self._deny(base_exec, args, cwd, path_err) if path_err else self._allow(base_exec, args, cwd)
        if "-c" in args or "-i" in args:
            return self._deny(base_exec, args, cwd, "Inline arbitrary code execution ('python -c') and interactive mode are not permitted.")
        if args in (["--version"], ["-V"]):
            return self._allow(base_exec, args, cwd)
        if len(args) >= 2 and args[0] == "-m":
            mod, mod_args = args[1], args[2:]
            if mod in {"pytest", "unittest", "py_compile", "compileall", "json.tool"}:
                err = self._validate_path_args(mod_args, cwd)
                return self._deny(base_exec, args, cwd, err) if err else self._allow(base_exec, args, cwd)
            if mod == "pip":
                sub = next((a for a in mod_args if not a.startswith("-")), "")
                if sub in {"list", "show", "check", "freeze"} or "--version" in mod_args:
                    return self._allow(base_exec, args, cwd)
                if sub in {"install", "uninstall"}:
                    return self._allow(base_exec, args, cwd, "destructive")
        return self._deny(base_exec, args, cwd, "Only '--version' and approved '-m' modules (pytest, unittest, py_compile, pip) are permitted for python.")

    def _evaluate_ros2_policy(self, base_exec: str, args: List[str], cwd: Path) -> PolicyDecision:
        if not args or args[0] in {"--help", "-h", "doctor", "wtf"}:
            return self._allow(base_exec, args, cwd)

        subcmd = args[0].lower()
        rest = args[1:]
        verb = next((a.lower() for a in rest if not a.startswith("-")), "")

        if subcmd in _ROS2_READONLY_VERBS and (verb in _ROS2_READONLY_VERBS[subcmd] or "--help" in rest or "-h" in rest):
            safe = list(args)
            if subcmd == "topic" and verb == "echo" and "--once" not in safe:
                safe.append("--once")
            return self._allow(base_exec, safe, cwd)

        if subcmd in {"run", "launch"} or (subcmd in _ROS2_DESTRUCTIVE_VERBS and verb in _ROS2_DESTRUCTIVE_VERBS[subcmd]):
            return self._allow(base_exec, args, cwd, "destructive")

        return self._deny(base_exec, args, cwd, f"ROS 2 command 'ros2 {subcmd} {verb}'.strip() is not permitted by policy.")


# ---------------------------------------------------------------------------
# Capability Tool Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Binds JARVIS capability tools (execute_command, filesystem, desktop, memory, calculate) with policy validation."""

    def __init__(
        self,
        memory_store: FAISSMemoryStore,
        workspace_root: Optional[Path] = None,
        workspace_roots: Optional[Sequence[Path]] = None,
        allow_app_launch: bool = True,
        command_timeout_seconds: int = 10,
        max_command_timeout_seconds: int = 30,
        max_output_chars: int = 4000,
        require_confirmation_for_destructive: bool = True,
    ):
        self.memory_store = memory_store
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        roots = tuple(Path(r).resolve() for r in workspace_roots) if workspace_roots else (self.workspace_root,)
        self.workspace_roots: Tuple[Path, ...] = roots if self.workspace_root in roots else (self.workspace_root, *roots)

        self.allow_app_launch = allow_app_launch
        self.command_timeout_seconds = command_timeout_seconds
        self.max_command_timeout_seconds = max_command_timeout_seconds
        self.max_output_chars = max_output_chars
        self.require_confirmation_for_destructive = require_confirmation_for_destructive

        self.policy = CommandPolicyValidator(
            workspace_roots=self.workspace_roots,
            require_confirmation_for_destructive=self.require_confirmation_for_destructive,
        )

        self.functions: Dict[str, Callable[..., Any]] = {
            "execute_command": self.execute_command,
            "filesystem": self.filesystem,
            "desktop": self.desktop,
            "memory": self.memory,
            "calculate": calculate,
        }

        self.schemas: List[Dict[str, Any]] = [
            {
                "name": "execute_command",
                "description": "Run a validated system, hardware, date/time, Git, Python/pytest, or ROS 2 CLI command using executable and args (no shell strings/pipes).",
                "arguments": {
                    "executable": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"}, "purpose": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": self.max_command_timeout_seconds},
                    "confirm": {"type": "boolean"},
                },
                "required": ["executable"],
            },
            {
                "name": "filesystem",
                "description": "Workspace-scoped file operations: list, search, read, write, edit, mkdir, copy, move, delete inside approved workspace roots.",
                "arguments": {
                    "operation": {"type": "string", "enum": ["list", "search", "read", "write", "edit", "mkdir", "copy", "move", "delete"]},
                    "path": {"type": "string"}, "destination": {"type": "string"}, "pattern": {"type": "string"},
                    "content": {"type": "string"}, "target": {"type": "string"}, "replacement": {"type": "string"},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 500},
                    "overwrite": {"type": "boolean"}, "confirm": {"type": "boolean"},
                },
                "required": ["operation"],
            },
            {
                "name": "desktop",
                "description": "Desktop control: launch approved GUI apps, list available desktop apps, or capture a desktop screenshot.",
                "arguments": {
                    "operation": {"type": "string", "enum": ["open_app", "list_apps", "screenshot"]},
                    "app_name": {"type": "string"}, "path": {"type": "string"},
                },
                "required": ["operation"],
            },
            {
                "name": "memory",
                "description": "Personal episodic, semantic, preference, procedural, or goal memory store (remember, search, update, forget, list).",
                "arguments": {
                    "operation": {"type": "string", "enum": ["remember", "search", "update", "forget", "list"]},
                    "text": {"type": "string"}, "query": {"type": "string"}, "memory_id": {"type": "integer"},
                    "memory_type": {"type": "string", "enum": list(VALID_MEMORY_TYPES)},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["operation"],
            },
            {
                "name": "calculate",
                "description": "Safely evaluate arithmetic or math expressions (with bounded exponents).",
                "arguments": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        ]
        self.schema_map: Dict[str, Dict[str, Any]] = {s["name"]: s for s in self.schemas}

    @property
    def tool_names(self) -> set[str]:
        return set(self.schema_map.keys())

    def format_schemas_for_prompt(self) -> str:
        return "\n".join(json.dumps(s, ensure_ascii=False) for s in self.schemas)

    def _build_sanitized_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for k, v in os.environ.items():
            upper = k.upper()
            if any(s in upper for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY")):
                continue
            if k in _SAFE_ENV_KEYS or upper.startswith(("ROS_", "LC_", "XDG_")):
                env[k] = v
        env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
        env["GIT_TERMINAL_PROMPT"], env["PAGER"] = "0", "cat"
        return env

    def _truncate_output(self, text: str) -> Tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        return f"{text[: self.max_output_chars]}\n...[truncated to {self.max_output_chars} chars]", True

    def execute_command(
        self,
        executable: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        purpose: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Validate and execute a structured command with `shell=False`."""
        effective_timeout = max(1, min(int(timeout_seconds or self.command_timeout_seconds), self.max_command_timeout_seconds))
        req = CommandRequest(executable=executable, args=list(args or []), cwd=cwd, purpose=purpose, timeout_seconds=effective_timeout, confirm=bool(confirm))
        decision = self.policy.validate(req)
        cmd_list = [decision.executable or executable, *(decision.args or req.args)]

        if not decision.allowed:
            return CommandResult(ok=False, status="denied", command=cmd_list, exit_code=None, stdout="", stderr="", duration_ms=0.0, permission_level=decision.permission_level, error=decision.reason).to_dict()

        if decision.requires_confirmation:
            return CommandResult(
                ok=False, status="confirmation_required", command=cmd_list, exit_code=None, stdout="", stderr="", duration_ms=0.0,
                permission_level=decision.permission_level, requires_confirmation=True,
                error=f"Command '{' '.join(cmd_list)}' is classified as {decision.permission_level} and requires explicit user confirmation before execution.",
            ).to_dict()

        resolved_bin = shutil.which(decision.executable)
        if not resolved_bin:
            return CommandResult(ok=False, status="missing_executable", command=cmd_list, exit_code=None, stdout="", stderr="", duration_ms=0.0, permission_level=decision.permission_level, error=f"Utility '{decision.executable}' is not installed or not found on PATH.").to_dict()

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [resolved_bin, *decision.args],
                cwd=str(decision.cwd or self.workspace_root),
                env=self._build_sanitized_env(),
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=effective_timeout,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            stdout, trunc_out = self._truncate_output(proc.stdout.strip())
            stderr, trunc_err = self._truncate_output(proc.stderr.strip())
            ok = proc.returncode == 0
            return CommandResult(
                ok=ok, status="completed" if ok else "failed", command=cmd_list, exit_code=proc.returncode,
                stdout=stdout, stderr=stderr, duration_ms=elapsed_ms, timed_out=False, truncated=trunc_out or trunc_err,
                permission_level=decision.permission_level, error=None if ok else (stderr or f"Command exited with code {proc.returncode}."),
            ).to_dict()
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            raw_out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            raw_err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stdout, trunc_out = self._truncate_output(raw_out.strip())
            stderr, trunc_err = self._truncate_output(raw_err.strip())
            return CommandResult(
                ok=False, status="timed_out", command=cmd_list, exit_code=None, stdout=stdout, stderr=stderr,
                duration_ms=elapsed_ms, timed_out=True, truncated=trunc_out or trunc_err, permission_level=decision.permission_level,
                error=f"Command timed out after {effective_timeout} seconds.",
            ).to_dict()
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return CommandResult(
                ok=False, status="failed", command=cmd_list, exit_code=None, stdout="", stderr="",
                duration_ms=elapsed_ms, permission_level=decision.permission_level, error=str(exc),
            ).to_dict()

    # -----------------------------------------------------------------------
    # Workspace-Scoped Filesystem Capability Tool
    # -----------------------------------------------------------------------

    def _resolve_safe_path(self, rel_path: str = ".") -> Path:
        return self.policy.resolve_safe_path(rel_path, base_dir=self.workspace_root)

    def filesystem(
        self,
        operation: str,
        path: str = ".",
        destination: Optional[str] = None,
        pattern: Optional[str] = None,
        content: Optional[str] = None,
        target: Optional[str] = None,
        replacement: Optional[str] = None,
        max_lines: int = 100,
        overwrite: bool = False,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Execute workspace-scoped file and directory operations."""
        try:
            resolved = self._resolve_safe_path(path)
        except PermissionError as exc:
            return {"ok": False, "status": "denied", "error": str(exc)}

        if operation == "list":
            if not resolved.exists() or not resolved.is_dir():
                return {"ok": False, "status": "failed", "error": f"Directory '{path}' does not exist."}
            entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            items = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:100]]
            return {"ok": True, "status": "completed", "path": path, "entries": items, "summary": ", ".join(items) if items else "(empty directory)"}

        if operation == "search":
            if not resolved.exists() or not resolved.is_dir():
                return {"ok": False, "status": "failed", "error": f"Directory '{path}' does not exist."}
            search_pat, matches, skip_dirs = (pattern or "*").strip(), [], {".git", "__pycache__", "node_modules", ".venv", "jarvis_data"}
            for root_dir, dirs, files in os.walk(resolved):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                rel_root = Path(root_dir).relative_to(self.workspace_root)
                for fname in sorted(files):
                    rel_file = str(rel_root / fname) if str(rel_root) != "." else fname
                    if fnmatch.fnmatch(fname.lower(), search_pat.lower()) or search_pat.lower() in rel_file.lower():
                        matches.append(rel_file)
                        if len(matches) >= 50: break
                if len(matches) >= 50: break
            return {"ok": True, "status": "completed", "pattern": search_pat, "matches": matches}

        if operation == "read":
            if not resolved.exists() or not resolved.is_file():
                return {"ok": False, "status": "failed", "error": f"File '{path}' does not exist."}
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
            limit = max(1, min(int(max_lines), 500))
            return {"ok": True, "status": "completed", "path": path, "content": "\n".join(lines[:limit]), "total_lines": len(lines), "truncated": len(lines) > limit}

        if operation == "write":
            if content is None:
                return {"ok": False, "status": "failed", "error": "'content' is required for filesystem write."}
            if resolved.exists() and not (overwrite or confirm):
                return {"ok": False, "status": "confirmation_required", "requires_confirmation": True, "error": f"File '{path}' already exists. Set overwrite=True or confirm=True to overwrite."}
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return {"ok": True, "status": "completed", "path": path, "bytes_written": len(content.encode("utf-8"))}

        if operation == "edit":
            if not resolved.exists() or not resolved.is_file():
                return {"ok": False, "status": "failed", "error": f"File '{path}' does not exist."}
            if not target:
                return {"ok": False, "status": "failed", "error": "'target' string is required for filesystem edit."}
            text = resolved.read_text(encoding="utf-8")
            if target not in text:
                return {"ok": False, "status": "failed", "error": f"Target string not found in '{path}'."}
            resolved.write_text(text.replace(target, replacement or ""), encoding="utf-8")
            return {"ok": True, "status": "completed", "path": path}

        if operation == "mkdir":
            resolved.mkdir(parents=True, exist_ok=True)
            return {"ok": True, "status": "completed", "path": path}

        if operation in {"copy", "move"}:
            if not destination:
                return {"ok": False, "status": "failed", "error": f"'destination' is required for filesystem {operation}."}
            try:
                dest_resolved = self._resolve_safe_path(destination)
            except PermissionError as exc:
                return {"ok": False, "status": "denied", "error": str(exc)}
            if not resolved.exists():
                return {"ok": False, "status": "failed", "error": f"Source path '{path}' does not exist."}
            dest_resolved.parent.mkdir(parents=True, exist_ok=True)
            if operation == "copy":
                shutil.copy2(resolved, dest_resolved)
            else:
                shutil.move(str(resolved), str(dest_resolved))
            return {"ok": True, "status": "completed", "source": path, "destination": destination}

        if operation == "delete":
            if resolved in self.workspace_roots:
                return {"ok": False, "status": "denied", "error": "Deleting a workspace root directory is forbidden."}
            if self.require_confirmation_for_destructive and not confirm:
                return {"ok": False, "status": "confirmation_required", "requires_confirmation": True, "error": f"Deleting '{path}' is destructive and requires explicit user confirmation (confirm=True)."}
            if not resolved.exists():
                return {"ok": False, "status": "failed", "error": f"Path '{path}' does not exist."}
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            return {"ok": True, "status": "completed", "deleted": path}

        return {"ok": False, "status": "failed", "error": f"Unsupported filesystem operation: {operation}"}

    # -----------------------------------------------------------------------
    # Desktop Capability Tool
    # -----------------------------------------------------------------------

    def desktop(self, operation: str, app_name: Optional[str] = None, path: str = "screenshot.png") -> Dict[str, Any]:
        """Open approved desktop applications, list apps, or capture screenshots."""
        system = platform.system().lower()
        commands = APP_COMMANDS.get(system, {})

        if operation == "list_apps":
            return {"ok": True, "status": "completed", "apps": {name: bool(shutil.which(cmd[0])) for name, cmd in commands.items()}}

        if operation == "open_app":
            if not app_name:
                return {"ok": False, "status": "failed", "error": "'app_name' is required for desktop open_app."}
            if not commands:
                return {"ok": False, "status": "failed", "error": f"Unsupported OS: {platform.system()}"}
            clean_name = re.sub(r"^(?:open|launch)\s+", "", app_name.strip(), flags=re.I).lower()
            cmd = commands.get(clean_name)
            if not cmd:
                return {"ok": False, "status": "denied", "error": f"No approved desktop application configured for '{app_name}'."}
            if not self.allow_app_launch:
                return {"ok": True, "status": "completed", "message": f"[Dry-run] Would launch {' '.join(cmd)}."}
            if not shutil.which(cmd[0]):
                return {"ok": False, "status": "missing_executable", "error": f"Application '{clean_name}' ('{cmd[0]}') is not installed or not on PATH."}
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
                return {"ok": True, "status": "completed", "message": f"Opening {clean_name}."}
            except Exception as exc:
                return {"ok": False, "status": "failed", "error": f"Could not open '{clean_name}': {exc}"}

        if operation == "screenshot":
            try:
                target = self._resolve_safe_path(path)
            except PermissionError as exc:
                return {"ok": False, "status": "denied", "error": str(exc)}
            if not self.allow_app_launch:
                return {"ok": True, "status": "completed", "message": f"[Dry-run] Would save screenshot to {target.name}."}
            candidates = [
                (["gnome-screenshot", "-f", str(target)], "gnome-screenshot"),
                (["scrot", str(target)], "scrot"),
                (["screencapture", "-x", str(target)], "screencapture"),
            ]
            for cmd, bin_name in candidates:
                if shutil.which(bin_name):
                    proc = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=10)
                    if proc.returncode == 0 and target.exists():
                        return {"ok": True, "status": "completed", "path": str(target.relative_to(self.workspace_root))}
            return {"ok": False, "status": "missing_executable", "error": "No supported screenshot utility (gnome-screenshot, scrot, screencapture) is available."}

        return {"ok": False, "status": "failed", "error": f"Unsupported desktop operation: {operation}"}

    # -----------------------------------------------------------------------
    # Consolidated Memory Capability Tool
    # -----------------------------------------------------------------------

    def memory(
        self,
        operation: str,
        text: Optional[str] = None,
        query: Optional[str] = None,
        memory_id: Optional[int] = None,
        memory_type: Optional[str] = None,
        importance: Optional[float] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Unified memory capability handler backed by FAISSMemoryStore."""
        if operation == "remember":
            if not text or not text.strip():
                return {"ok": False, "status": "failed", "error": "'text' is required for memory operation 'remember'."}
            mid = self.memory_store.add(
                text=text.strip(), memory_type=memory_type or "semantic",
                importance=0.7 if importance is None else float(importance), source="tool_call",
            )
            return {"ok": True, "status": "completed", "message": f"Memory stored with id {mid}."}

        if operation == "search":
            if not query or not query.strip():
                return {"ok": False, "status": "failed", "error": "'query' is required for memory operation 'search'."}
            results = self.memory_store.hybrid_search(query=query.strip(), top_k=top_k, memory_type=memory_type, query_aware=False)
            lines = [f"[id={r['id']} type={r['memory_type']} score={r['score']:.3f}] {r['text']}" for r in results]
            return {"ok": True, "status": "completed", "results": "\n".join(lines) if lines else "No relevant memories found."}

        if operation == "update":
            if memory_id is None or not text or not text.strip():
                return {"ok": False, "status": "failed", "error": "'memory_id' and 'text' are required for memory operation 'update'."}
            ok = self.memory_store.update(memory_id=int(memory_id), text=text.strip(), memory_type=memory_type, importance=importance, source="tool_update")
            return {"ok": ok, "status": "completed" if ok else "failed", "message": f"Memory {memory_id} updated." if ok else f"Memory id {memory_id} not found."}

        if operation == "forget":
            if memory_id is None:
                return {"ok": False, "status": "failed", "error": "'memory_id' is required for memory operation 'forget'."}
            ok = self.memory_store.delete(int(memory_id))
            return {"ok": ok, "status": "completed" if ok else "failed", "message": f"Memory {memory_id} deleted." if ok else f"Memory id {memory_id} not found."}

        if operation == "list":
            items = self.memory_store.list_all()
            if memory_type:
                items = [m for m in items if m["memory_type"] == memory_type]
            return {"ok": True, "status": "completed", "memories": items[: max(1, min(int(top_k), 50))]}

        return {"ok": False, "status": "failed", "error": f"Unsupported memory operation: {operation}"}

    # -----------------------------------------------------------------------
    # Schema Validation & Execution Dispatch
    # -----------------------------------------------------------------------

    def validate_tool_arguments(self, tool_name: str, arguments: Any) -> Tuple[bool, Optional[str]]:
        if tool_name not in self.schema_map:
            return False, f"Unknown tool: {tool_name}"
        if not isinstance(arguments, dict):
            return False, "Tool arguments must be a JSON object."

        schema = self.schema_map[tool_name]
        for req in schema.get("required", []):
            if req not in arguments or arguments[req] is None:
                return False, f"Missing required argument '{req}' for tool '{tool_name}'."

        prop_specs = schema.get("arguments", {})
        type_checkers = {
            "string": lambda v: isinstance(v, str),
            "boolean": lambda v: isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
        }

        for arg_name, arg_val in arguments.items():
            if arg_name not in prop_specs:
                return False, f"Unexpected argument '{arg_name}' for tool '{tool_name}'."
            if arg_val is None:
                continue
            spec = prop_specs[arg_name]
            expected_type = spec.get("type")
            checker = type_checkers.get(expected_type)
            if checker and not checker(arg_val):
                return False, f"Argument '{arg_name}' must be a {expected_type}."
            if expected_type == "array" and spec.get("items", {}).get("type") == "string":
                if any(not isinstance(x, str) for x in arg_val):
                    return False, f"All elements of '{arg_name}' must be strings."
            if "enum" in spec and arg_val not in spec["enum"]:
                return False, f"Argument '{arg_name}' must be one of {spec['enum']}."
            if isinstance(arg_val, (int, float)) and not isinstance(arg_val, bool):
                if "minimum" in spec and arg_val < spec["minimum"]:
                    return False, f"Argument '{arg_name}' must be >= {spec['minimum']}."
                if "maximum" in spec and arg_val > spec["maximum"]:
                    return False, f"Argument '{arg_name}' must be <= {spec['maximum']}."

        return True, None

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        valid, err = self.validate_tool_arguments(tool_name, arguments)
        if not valid:
            return {"ok": False, "status": "invalid_arguments", "error": err}
        fn = self.functions[tool_name]
        try:
            clean_args = {k: v for k, v in arguments.items() if v is not None}
            result = fn(**clean_args)
            if isinstance(result, dict) and "ok" in result:
                return {"ok": bool(result["ok"]), "result": result, **result}
            return {"ok": True, "status": "completed", "result": result}
        except Exception as exc:
            return {"ok": False, "status": "failed", "error": str(exc)}
