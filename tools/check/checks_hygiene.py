#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from __future__ import annotations

"""
Hardcoding, residual-system, modality and engine-isolation conformance checks.

Covers ``docs/prompts/UPGRADE_05_CLEANUP.md``, ``UPGRADE_03_ARCH.md`` §3.3 and
the engine red line in ``UPGRADE_04_ALGO.md`` §4.5.
"""

import ast
import re
from pathlib import Path

from tools.check.base import (
    PiscesLxAstWalker,
    PiscesLxRegexLibrary,
    PiscesLxRepositoryScanner,
    PiscesLxSeverity,
    PiscesLxSourceFile,
    PiscesLxStandardCheck,
    PiscesLxViolation,
)

__all__ = [
    "PiscesLxHardcodingCheck",
    "PiscesLxResidualSystemCheck",
    "PiscesLxModalityCheck",
    "PiscesLxEngineIsolationCheck",
]


class PiscesLxHardcodingCheck(PiscesLxStandardCheck):
    """§10.2(13) — paths, devices, model identifiers and magic numbers come from config."""

    rule = "NO-HARDCODED-PATH"
    description = "no hardcoded paths / devices / model ids / unexplained numeric literals"

    _MODEL_ID_PATTERNS = (
        re.compile(r"[\"'][A-Za-z0-9_.-]+/[A-Za-z0-9_.-]*(?:Qwen|Llama|DeepSeek|ChatGLM|Baichuan|Gemma|Mistral|Phi)[A-Za-z0-9_.-]*[\"']", re.IGNORECASE),
    )
    _STRUCTURAL = frozenset({"0", "1", "-1", "2", "0.0", "1.0", "1e-8", "1e-6", "2.0", "0.5"})
    _ALLOWED_ARG_NAMES = re.compile(r"^(?:[-+]?\d+(?:\.\d+)?[eE][-+]?\d+|nan|inf)$")

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Run literal scanning and magic-number AST analysis."""
        violations = []
        for source in scanner.files:
            violations.extend(self._scan_literals(source))
            violations.extend(self._scan_magic_numbers(source))
        return violations

    def _relative(self, source: PiscesLxSourceFile) -> str:
        return source.path.relative_to(source.root).as_posix()

    def _scan_literals(self, source: PiscesLxSourceFile) -> list:
        violations = []
        relative = self._relative(source)
        if relative.startswith(("configs/version.py", "utils/paths.py")):
            allowlist_root = True
        else:
            allowlist_root = False
        for index, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern in PiscesLxRegexLibrary.absolute_path_patterns():
                match = pattern.search(line)
                if match:
                    if allowlist_root and "bootstrap" in line:
                        continue
                    violations.append(
                        PiscesLxViolation(
                            self.rule,
                            source.path,
                            index,
                            f"hardcoded filesystem path `{match.group(0)}` — resolve through "
                            "configs or environment variables",
                            snippet=stripped[:160],
                        )
                    )
                    break
            match = PiscesLxRegexLibrary.CUDA_DEVICE.search(line)
            if match:
                violations.append(
                    PiscesLxViolation(
                        "NO-HARDCODED-DEVICE",
                        source.path,
                        index,
                        f"hardcoded device literal `{match.group(0)}` — select the device from "
                        "configuration, never pin a GPU index",
                        snippet=stripped[:160],
                    )
                )
            match = PiscesLxRegexLibrary.MULTI_GPU_DEVICE.search(line)
            if match:
                violations.append(
                    PiscesLxViolation(
                        "NO-HARDCODED-DEVICE",
                        source.path,
                        index,
                        f"hardcoded accelerator index `{match.group(0)}`",
                        snippet=stripped[:160],
                    )
                )
            for pattern in self._MODEL_ID_PATTERNS:
                match = pattern.search(line)
                if match:
                    violations.append(
                        PiscesLxViolation(
                            "NO-HARDCODED-MODEL-ID",
                            source.path,
                            index,
                            f"third-party model identifier `{match.group(0)}` baked into source — "
                            "model/dataset names come from config",
                            snippet=stripped[:160],
                        )
                    )
                    break
        return violations

    def _scan_magic_numbers(self, source: PiscesLxSourceFile) -> list:
        tree = source.tree
        if tree is None:
            return []
        violations = []
        parents = PiscesLxAstWalker.enclosing_classes(tree)
        constant_parents = self._constant_assignment_targets(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            value = node.value
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            rendering = self._render(value)
            if rendering in self._STRUCTURAL or self._ALLOWED_ARG_NAMES.match(rendering):
                continue
            if node in constant_parents:
                continue
            if self._is_annotation_value(node, parents):
                continue
            if self._inside_named_constant(node, parents):
                continue
            line = source.line_source(node.lineno)
            if "#" in line:
                continue
            if not self._is_computational(node, parents):
                continue
            violations.append(
                PiscesLxViolation(
                    "NO-MAGIC-NUMBER",
                    source.path,
                    node.lineno,
                    f"unexplained numeric literal `{rendering}` — bind it to a named constant "
                    "sourced from configuration, or comment the reason it is fixed",
                    column=getattr(node, "col_offset", 0),
                    severity=PiscesLxSeverity.ERROR,
                    snippet=line,
                )
            )
        return violations

    @staticmethod
    def _render(value) -> str:
        if isinstance(value, float):
            text = repr(value)
        else:
            text = str(value)
        if text.startswith("+"):
            text = text[1:]
        return text

    @staticmethod
    def _constant_assignment_targets(tree: ast.Module) -> set:
        sinks = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        sinks.add(node.value)
                        if isinstance(node.value, ast.Constant):
                            sinks.add(node.value)
        return sinks

    @staticmethod
    def _inside_named_constant(node: ast.AST, parents: dict) -> bool:
        current = parents.get(node)
        depth = 0
        while current is not None and depth < 3:
            if isinstance(current, ast.Assign):
                for target in current.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        return True
            current = parents.get(current)
            depth += 1
        return False

    @staticmethod
    def _is_annotation_value(node: ast.AST, parents: dict) -> bool:
        current = parents.get(node)
        while current is not None:
            if isinstance(current, (ast.arg, ast.AnnAssign)):
                return True
            current = parents.get(current)
        return False

    @staticmethod
    def _is_computational(node: ast.AST, parents: dict) -> bool:
        """Only flag literals that participate in computation or configuration values."""
        parent = parents.get(node)
        if parent is None:
            return False
        if isinstance(parent, ast.Constant):
            return False
        if isinstance(parent, ast.Subscript):
            return parent.slice is node
        if isinstance(parent, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
            return True
        if isinstance(parent, ast.Call):
            return True
        if isinstance(parent, ast.keyword):
            return True
        if isinstance(parent, ast.Dict):
            return True
        if isinstance(parent, ast.Return):
            return True
        if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
            return True
        if isinstance(parent, ast.If):
            return True
        return False

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Unused — :meth:`run` iterates the corpus directly."""
        del source
        return []


class PiscesLxResidualSystemCheck(PiscesLxStandardCheck):
    """§05 — EnTA/Encre deleted, watermark and MCP service layer relocated off-model."""

    rule = "CLEAN-ENTA-ENCRE"
    description = "no EnTA/Encre/watermark/service-layer references remain in model code"

    _ENTA = (
        re.compile(r"\bEnta[A-Z]\w*"),
        re.compile(r"\bENTA\b"),
        re.compile(r"\benta_[a-z_]+\b"),
        re.compile(r"\benable_enta\b"),
        re.compile(r"\bEncre\b"),
        re.compile(r"agentevolver"),
        re.compile(r"teachers\.yaml"),
        re.compile(r"model\.agentic\.enta"),
    )
    _WATERMARK = (
        re.compile(r"\bwatermark\w*\b", re.IGNORECASE),
        re.compile(r"\bWMC\b"),
        re.compile(r"opss\.watermark"),
        re.compile(r"tools\.wmc"),
    )
    _MCP_SERVICE = (
        re.compile(r"\bopss\.mcp\b"),
        re.compile(r"\bfrom opss import mcp\b"),
        re.compile(r"\bMcpServer\b"),
        re.compile(r"\bmcp_server\b"),
    )

    _EXEMPT_ROOTS = ("third_party/", "docs/", ".git/", "tools/check/")

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Report residual identifiers, skipping relocated third-party and doc trees."""
        violations = []
        for source in scanner.files:
            relative = source.path.relative_to(source.root).as_posix()
            if relative.startswith(self._EXEMPT_ROOTS):
                continue
            if relative.startswith(("enta/", "model/agentic/enta/", "opss/watermark/", "tools/wmc/", "opss/mcp/")):
                continue
            violations.extend(self._scan(source, self._ENTA, "CLEAN-ENTA-ENCRE",
                                         "EnTA/Encre reference survives — the training agent system is deleted"))
            violations.extend(self._scan(source, self._WATERMARK, "CLEAN-WATERMARK",
                                         "watermark reference survives in model code — it belongs to the third-party tier"))
            violations.extend(self._scan(source, self._MCP_SERVICE, "CLEAN-MCP-SERVICE-LAYER",
                                         "MCP service-layer reference survives — move it out of the model tree"))
        return violations

    @staticmethod
    def _scan(source: PiscesLxSourceFile, patterns: tuple, rule: str, message: str) -> list:
        hits = []
        for index, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    hits.append(
                        PiscesLxViolation(
                            rule,
                            source.path,
                            index,
                            f"{message} (matched `{match.group(0)}`)",
                            snippet=stripped[:160],
                        )
                    )
                    break
        return hits

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Unused — :meth:`run` drives the scan."""
        del source
        return []


class PiscesLxModalityCheck(PiscesLxStandardCheck):
    """§3.3.1 — modalities are exactly text/image/audio/video; document and agent are gone."""

    rule = "MODALITY-4-NO-DOCUMENT"
    description = "document/agent must not survive as independent modalities"

    _DOC = (
        re.compile(r"\bYvDoc\w*"),
        re.compile(r"\bYvDocument\w*"),
        re.compile(r"\bdoc_encoder\b"),
        re.compile(r"\bdocument_encoder\b"),
        re.compile(r"\buse_doc\b"),
        re.compile(r"\bdoc_modality\b"),
        re.compile(r"\bModality(?:Type)?\.DOCUMENT\b"),
        re.compile(r"[\"']document[\"']\s*[,\]]\s*[\"'](?:agent|text|image)"),
        re.compile(r"LayoutLM\w*"),
    )
    _AGENT_MODALITY = (
        re.compile(r"\bYvAgenticEncoder\b"),
        re.compile(r"\bagentic_encoder\b"),
        re.compile(r"\bagent_modality\b"),
        re.compile(r"\buse_agentic_modality\b"),
        re.compile(r"\bModality(?:Type)?\.AGENT\b"),
        re.compile(r"\bmodal_type\b\s*==\s*[\"']agent[\"']"),
    )
    _EXPECTED = ("text", "image", "audio", "video")

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Flag residual document/agent modality encoders and identifiers."""
        violations = []
        for source in scanner.files:
            relative = source.path.relative_to(source.root).as_posix()
            if relative.startswith(self._EXEMPT):
                continue
            violations.extend(
                self._scan(source, self._DOC, "MODALITY-4-NO-DOCUMENT",
                           "document survives as an independent modality — route it through agent+text")
            )
            violations.extend(
                self._scan(source, self._AGENT_MODALITY, "MODALITY-4-NO-AGENT",
                           "agent survives as an independent modality — it is text-level tool calling")
            )
        return violations

    _EXEMPT = ("third_party/", "docs/", "tools/check/")

    @staticmethod
    def _scan(source: PiscesLxSourceFile, patterns: tuple, rule: str, message: str) -> list:
        hits = []
        for index, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    hits.append(
                        PiscesLxViolation(
                            rule,
                            source.path,
                            index,
                            f"{message} (matched `{match.group(0)}`)",
                            snippet=stripped[:160],
                        )
                    )
                    break
        return hits

    @classmethod
    def supported_modalities(cls) -> tuple:
        """The only four modalities the upgraded framework accepts."""
        return cls._EXPECTED

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Unused — :meth:`run` drives the scan."""
        del source
        return []


class PiscesLxDanglingImportCheck(PiscesLxStandardCheck):
    """§05.4 / §06.8 — no import may point at a module that no longer exists."""

    rule = "NO-DANGLING-IMPORT"
    description = "every first-party import resolves to a file or package on disk"

    _FIRST_PARTY = ("model", "opss", "tools", "utils", "configs", "enta")

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Resolve every first-party import, absolute or relative, against the tree."""
        violations = []
        for source in scanner.files:
            violations.extend(self._inspect_file(source))
        return violations

    def _inspect_file(self, source: PiscesLxSourceFile) -> list:
        tree = source.tree
        if tree is None:
            return []
        violations = []
        # parts[:-1] already drops the file name; taking .parent as well would strip the
        # enclosing package too, mis-resolving `from .x` imports one level too high.
        package_parts = source.path.relative_to(source.root).parts[:-1]
        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self._FIRST_PARTY:
                        target = alias.name
                        if not self._resolves(source.root, target):
                            violations.append(self._violation(source, node, target))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package_parts[: len(package_parts) - node.level + 1]
                    if node.module:
                        base = base + tuple(node.module.split("."))
                    target = ".".join(part for part in base if part)
                    if target.split(".")[0] in self._FIRST_PARTY and not self._resolves(
                        source.root, target
                    ):
                        violations.append(self._violation(source, node, target))
                elif node.module and node.module.split(".")[0] in self._FIRST_PARTY:
                    target = node.module
                    if not self._resolves(source.root, target):
                        violations.append(self._violation(source, node, target))
        return violations

    @staticmethod
    def _resolves(root: Path, dotted: str) -> bool:
        """Walk every component: a name resolves only if the *full* path exists.

        An earlier version short-circuited with ``... or True`` on the first component,
        which made the whole check vacuous — it reported zero dangling imports while real
        broken relative imports (``from .watermark import …`` after the watermark module
        was relocated) were still in the tree.
        """
        parts = dotted.split(".")
        if parts[0] not in ("model", "opss", "tools", "utils", "configs", "enta"):
            return True
        cursor = root
        for index, part in enumerate(parts):
            cursor = cursor / part
            if cursor.with_suffix(".py").is_file():
                # A module file: anything imported *out of* it is a symbol question,
                # not a path question, so the path itself resolves here.
                return index == len(parts) - 1 or (cursor.is_dir())
            if cursor.is_dir():
                continue
            return False
        return (cursor / "__init__.py").is_file() or cursor.is_file()

    @staticmethod
    def _violation(source: PiscesLxSourceFile, node: ast.AST, target: str) -> PiscesLxViolation:
        return PiscesLxViolation(
            "NO-DANGLING-IMPORT",
            source.path,
            getattr(node, "lineno", 1),
            f"import of '{target}' does not resolve — the module was deleted or renamed; "
            "no broken reference may remain in the tree",
            column=getattr(node, "col_offset", 0),
            snippet=source.line_source(getattr(node, "lineno", 1)),
        )

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Unused — :meth:`run` drives the resolution pass."""
        del source
        return []


class PiscesLxEngineIsolationCheck(PiscesLxStandardCheck):
    """§04.5 — training/inference engines never load, call or proxy an external model."""

    rule = "ENGINE-NO-EXTERNAL-MODEL"
    description = "engine code must not instantiate or base itself on a third-party model"

    _EXTERNAL_LOADERS = (
        re.compile(r"\bAuto(?:Model|ForCausalLM|Tokenizer|ForSeq2SeqLM|Processor)\b"),
        re.compile(r"\bfrom_pretrained\s*\("),
        re.compile(r"\bpipeline\s*\("),
        re.compile(r"\bGPT2LMHeadModel\b|\bLlamaForCausalLM\b|\bQwen2ForCausalLM\b|\bMistralForCausalLM\b"),
    )
    _EXTERNAL_NAME_LITERALS = re.compile(
        r"[\"'][^\"']*(?:Qwen|Llama|DeepSeek|ChatGLM|Baichuan|Gemma|Mistral|Phi-|Yi-|InternLM|Mixtral)[^\"']*[\"']",
        re.IGNORECASE,
    )
    _ENGINE_ROOTS = ("model/", "opss/train/", "opss/infer/", "opss/optim/", "opss/quantize/", "opss/quantization/")
    _OFFLINE_DISTILL_ROOTS = ("tools/data/", "opss/knowledge/", "scripts/distill/")

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Scan engine paths for external-model coupling."""
        violations = []
        for source in scanner.files:
            relative = source.path.relative_to(source.root).as_posix()
            if not relative.startswith(self._ENGINE_ROOTS):
                continue
            if relative.startswith(self._OFFLINE_DISTILL_ROOTS):
                continue
            if relative.startswith("docs/"):
                continue
            violations.extend(self._scan_file(source, relative))
        return violations

    def _scan_file(self, source: PiscesLxSourceFile, relative: str) -> list:
        """Flag **executable** coupling only: docstrings and comments are not couplings."""
        hits = []
        tree = source.tree
        if tree is None:
            return hits
        docstring_lines = self._docstring_lines(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "transformers" or alias.name.startswith("transformers."):
                        if node.lineno in docstring_lines:
                            continue
                        hits.append(
                            PiscesLxViolation(
                                self.rule, source.path, node.lineno,
                                "engine imports `transformers` model loaders — only PiscesL1's own "
                                "model classes may be instantiated inside the engine",
                                snippet=source.line_source(node.lineno),
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("transformers"):
                    imported = {a.name for a in node.names}
                    loaders = imported & self._LOADER_NAMES
                    if loaders:
                        hits.append(
                            PiscesLxViolation(
                                self.rule, source.path, node.lineno,
                                f"engine imports external loader(s) {sorted(loaders)} from "
                                f"'{node.module}' — forbidden inside model/train/infer engines",
                                snippet=source.line_source(node.lineno),
                            )
                        )
            elif isinstance(node, ast.Call):
                rendering = self._call_rendering(node)
                if not rendering:
                    continue
                if self._is_external_pretrained_call(node, rendering):
                    hits.append(
                        PiscesLxViolation(
                            self.rule, source.path, node.lineno,
                            f"engine instantiates an external model via `{rendering[:70]}`",
                            snippet=source.line_source(node.lineno),
                        )
                    )
        for index, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or index in docstring_lines:
                continue
            if stripped.startswith(('"""', "'''")) or '"""' in stripped and len(stripped) < 4:
                continue
            match = self._EXTERNAL_NAME_LITERALS.search(stripped)
            # A model *identifier* is a path/hub-id token; prose descriptions are not coupling.
            if match and not re.search(r"\bYv[A-Z]\w*", stripped) and " " not in match.group(0).strip("\"'"):
                hits.append(
                    PiscesLxViolation(
                        self.rule, source.path, index,
                        f"external model identifier `{match.group(0)[:60]}` in engine path '{relative}'",
                        snippet=stripped[:160],
                    )
                )
        return hits

    # Importing a *model* loader into the engine is forbidden outright. A vocabulary
    # reader (AutoTokenizer/AutoProcessor) is only constrained at its call site — see
    # _is_external_pretrained_call — because reading PiscesL1's own tokenizer.json through
    # it is library use, not a third-party model dependency.
    _LOADER_NAMES = frozenset({
        "AutoModel", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM",
        "AutoModelForCausalLMWithLMHead", "AutoImageProcessor",
    })

    @classmethod
    def _is_external_pretrained_call(cls, node: ast.Call, rendering: str) -> bool:
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in ("from_pretrained", "pipeline"):
            return False
        base = func.value
        base_name = getattr(base, "id", None) or getattr(getattr(base, "func", None), "attr", None)
        if base_name is None:
            return False
        if base_name in {"transformers"}:
            return True
        is_model_loader = base_name in cls._MODEL_LOADERS or base_name.startswith("Auto")
        if not is_model_loader:
            return False
        if base_name in cls._VOCAB_LOADERS:
            # A vocabulary reader is not a model, but only while it can never reach the Hub:
            # PiscesL1 must read its own tokenizer artefacts and nothing else.
            return not any(
                kw.arg == "local_files_only" and getattr(kw.value, "value", None) is True
                for kw in node.keywords
            )
        return True

    _MODEL_LOADERS = frozenset({"AutoModel", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM"})
    _VOCAB_LOADERS = frozenset({"AutoTokenizer", "AutoProcessor"})

    @staticmethod
    def _call_rendering(node: ast.Call) -> str:
        try:
            import ast as _ast
            return _ast.unparse(node)[:160]
        except Exception:
            return ""

    @staticmethod
    def _docstring_lines(tree: ast.Module) -> set:
        """Line numbers occupied by docstrings, so prose is never read as code."""
        occupied = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                occupied.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        return occupied

    def _scan_legacy(self, source: PiscesLxSourceFile, relative: str) -> list:
        hits = []
        for index, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("\"\"\"") or stripped.startswith("def "):
                continue
            if self._is_own_pretrain(source, stripped):
                continue
            for pattern in self._EXTERNAL_LOADERS:
                match = pattern.search(stripped)
                if match:
                    hits.append(
                        PiscesLxViolation(
                            self.rule, source.path, index,
                            f"engine touches an external model facility `{match.group(0)}` — "
                            "teachers may only appear in offline distillation data generation",
                            snippet=stripped[:160],
                        )
                    )
                    break
        return hits

    @staticmethod
    def _is_own_pretrain(source: PiscesLxSourceFile, stripped: str) -> bool:
        """PiscesL1's own checkpoint loading is legitimate, not an external model."""
        if "from_pretrained" not in stripped:
            return False
        return bool(
            re.search(r"PiscesL1|YvConfig|YvModel|self|cls|config|tokenizer|checkpoint|path", stripped)
        )

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Unused — :meth:`run` drives the scan."""
        del source
        return []
