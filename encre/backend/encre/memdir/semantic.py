#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}", text.lower())


def _build_idf(corpus: list[str]) -> dict[str, float]:
    n = len(corpus)
    if n == 0:
        return {}
    df: dict[str, int] = {}
    for doc in corpus:
        seen: set[str] = set()
        for token in _tokenize(doc):
            if token not in seen:
                df[token] = df.get(token, 0) + 1
                seen.add(token)
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def _tf_idf_vectorize(doc: str, idf: dict[str, float], vocabulary: set[str]) -> dict[str, float]:
    tokens = _tokenize(doc)
    if not tokens:
        return {}
    tf = Counter(tokens)
    total = len(tokens)
    out: dict[str, float] = {}
    for term, count in tf.items():
        if term in vocabulary:
            out[term] = (count / total) * idf.get(term, 1.0)
    return out


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard_similarity(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class SearchResult:
    file_name: str
    score: float
    snippet: str = ""
    memory_type: str = ""
    description: str = ""


class SemanticMemorySearch:
    """Pure local memory retrieval.

    This module intentionally does not initialize or download any separate
    embedding model. Retrieval is based on tf-idf with Jaccard fallback.
    Higher-level LLM reranking, if needed, should use the current session
    backend chosen by the user rather than an internal model.
    """

    def __init__(self, memory_dir: str) -> None:
        self._memory_dir = memory_dir
        self._corpus: dict[str, str] = {}
        self._idf: dict[str, float] = {}
        self._vocabulary: set[str] = set()
        self._dirty = True

    def index(self, files: dict[str, str]) -> None:
        self._corpus = dict(files)
        self._idf = _build_idf(list(files.values()))
        self._vocabulary = set(self._idf)
        self._dirty = False

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> list[SearchResult]:
        if self._dirty:
            self._rebuild_from_disk()
        if not self._corpus:
            return []

        results: list[SearchResult] = []
        q_vec = _tf_idf_vectorize(query, self._idf, self._vocabulary)
        if q_vec:
            for name, text in self._corpus.items():
                d_vec = _tf_idf_vectorize(text, self._idf, self._vocabulary)
                score = _cosine_similarity(q_vec, d_vec)
                if score >= min_score:
                    results.append(SearchResult(file_name=name, score=score, snippet=text[:200]))

        if not results:
            for name, text in self._corpus.items():
                score = _jaccard_similarity(query, text)
                if score >= min_score:
                    results.append(SearchResult(file_name=name, score=score, snippet=text[:200]))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def search_relevant(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.search(query, top_k=top_k, min_score=0.10)

    def _rebuild_from_disk(self) -> None:
        files: dict[str, str] = {}
        try:
            with os.scandir(self._memory_dir) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if entry.name == "MEMORY.md" or entry.name.startswith(".") or not entry.name.endswith(".md"):
                        continue
                    try:
                        with open(entry.path, "r", encoding="utf-8") as f:
                            files[entry.name] = f.read()
                    except (OSError, UnicodeDecodeError):
                        pass
        except OSError:
            pass
        self.index(files)


@dataclass
class WorkingMemory:
    current_goal: str = ""
    subgoals: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    scratchpad: list[str] = field(default_factory=list)
    _created_at: float = field(default_factory=time.time)

    def set_goal(self, goal: str) -> None:
        self.current_goal = goal

    def add_subgoal(self, subgoal: str) -> None:
        if subgoal not in self.subgoals:
            self.subgoals.append(subgoal)

    def complete_subgoal(self, subgoal: str) -> None:
        if subgoal in self.subgoals:
            self.subgoals.remove(subgoal)

    def add_hypothesis(self, hypothesis: str) -> None:
        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)

    def confirm_hypothesis(self, hypothesis: str) -> None:
        if hypothesis in self.hypotheses:
            self.hypotheses.remove(hypothesis)
        if hypothesis not in self.findings:
            self.findings.append(f"[CONFIRMED] {hypothesis}")

    def reject_hypothesis(self, hypothesis: str) -> None:
        if hypothesis in self.hypotheses:
            self.hypotheses.remove(hypothesis)
        self.findings.append(f"[REJECTED] {hypothesis}")

    def add_finding(self, finding: str) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def add_question(self, question: str) -> None:
        if question not in self.open_questions:
            self.open_questions.append(question)

    def resolve_question(self, question: str, answer: str = "") -> None:
        if question in self.open_questions:
            self.open_questions.remove(question)
        entry = f"Q: {question}"
        if answer:
            entry += f" -> {answer}"
        self.findings.append(entry)

    def note(self, text: str) -> None:
        self.scratchpad.append(text)

    def summarize(self) -> str:
        parts: list[str] = []
        if self.current_goal:
            parts.append(f"Goal: {self.current_goal}")
        if self.subgoals:
            parts.append("Subgoals:")
            parts.extend(f"  - {sg}" for sg in self.subgoals)
        if self.hypotheses:
            parts.append("Hypotheses:")
            parts.extend(f"  - {h}" for h in self.hypotheses)
        if self.findings:
            parts.append("Findings:")
            parts.extend(f"  - {f}" for f in self.findings[-10:])
        if self.open_questions:
            parts.append("Open questions:")
            parts.extend(f"  - {q}" for q in self.open_questions)
        if self.scratchpad:
            parts.append("Scratchpad:")
            parts.extend(f"  - {s}" for s in self.scratchpad[-5:])
        return "\n".join(parts) if parts else "(empty working memory)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_goal": self.current_goal,
            "subgoals": list(self.subgoals),
            "hypotheses": list(self.hypotheses),
            "findings": list(self.findings),
            "open_questions": list(self.open_questions),
            "scratchpad": list(self.scratchpad),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkingMemory":
        wm = cls()
        wm.current_goal = d.get("current_goal", "")
        wm.subgoals = d.get("subgoals", [])
        wm.hypotheses = d.get("hypotheses", [])
        wm.findings = d.get("findings", [])
        wm.open_questions = d.get("open_questions", [])
        wm.scratchpad = d.get("scratchpad", [])
        return wm


@dataclass
class ConsolidationAction:
    action: str
    file_a: str
    file_b: str = ""
    reason: str = ""
    merged_content: str = ""


class MemoryConsolidator:
    SIMILARITY_THRESHOLD = 0.75
    CONFLICT_PATTERNS = [
        (r"\b(?:do not|don't|never|should not|must not|avoid|forbidden|prohibited)\b",
         r"\b(?:do|always|should|must|use|prefer|recommended|allowed)\b"),
        (r"\b(?:remove|delete|drop|discard|abandon)\b",
         r"\b(?:keep|retain|preserve|maintain|add)\b"),
    ]

    def __init__(self, memory_dir: str) -> None:
        self._memory_dir = memory_dir

    def find_duplicates(self, files: dict[str, str]) -> list[ConsolidationAction]:
        actions: list[ConsolidationAction] = []
        names = list(files.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = _jaccard_similarity(files[names[i]], files[names[j]])
                if score >= self.SIMILARITY_THRESHOLD:
                    actions.append(ConsolidationAction(
                        action="merge",
                        file_a=names[i],
                        file_b=names[j],
                        reason=f"Jaccard similarity {score:.2f} >= {self.SIMILARITY_THRESHOLD}",
                        merged_content=self._merge_pair(files[names[i]], files[names[j]]),
                    ))
        return actions

    def find_conflicts(self, files: dict[str, str]) -> list[ConsolidationAction]:
        actions: list[ConsolidationAction] = []
        names = list(files.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = files[names[i]]
                b = files[names[j]]
                if _jaccard_similarity(a, b) < 0.30:
                    continue
                if self._has_opposing_claims(a, b):
                    actions.append(ConsolidationAction(
                        action="flag_conflict",
                        file_a=names[i],
                        file_b=names[j],
                        reason="Opposing claims detected",
                    ))
        return actions

    def find_stale(self, files: dict[str, str], age_days: dict[str, int], stale_threshold_days: int = 30) -> list[ConsolidationAction]:
        actions: list[ConsolidationAction] = []
        for name, text in files.items():
            days = age_days.get(name, 0)
            if days < stale_threshold_days:
                continue
            refs = re.findall(r"`([\w./-]+\.[\w]+):\d+`|`([\w./-]+\.[\w]+)`", text)
            missing: list[str] = []
            for ref_tuple in refs:
                path = ref_tuple[0] or ref_tuple[1]
                if path and not self._path_exists(path):
                    missing.append(path)
            if missing:
                actions.append(ConsolidationAction(
                    action="mark_stale",
                    file_a=name,
                    reason=f"References missing files: {', '.join(missing[:3])}",
                ))
        return actions

    def consolidate(self, files: dict[str, str], age_days: dict[str, int] | None = None) -> list[ConsolidationAction]:
        actions = self.find_duplicates(files) + self.find_conflicts(files)
        if age_days:
            actions.extend(self.find_stale(files, age_days))
        priority = {"merge": 0, "flag_conflict": 1, "mark_stale": 2}
        actions.sort(key=lambda a: priority.get(a.action, 99))
        return actions

    @staticmethod
    def _merge_pair(text_a: str, text_b: str) -> str:
        primary = text_a if len(text_a) >= len(text_b) else text_b
        secondary = text_b if primary == text_a else text_a
        pri_lines = set(primary.strip().split("\n"))
        unique = [l for l in secondary.strip().split("\n") if l not in pri_lines and len(l.strip()) > 10]
        if unique:
            primary += "\n\n## Merged from duplicate\n" + "\n".join(unique[:20])
        return primary

    def _has_opposing_claims(self, text_a: str, text_b: str) -> bool:
        for pos_pat, neg_pat in self.CONFLICT_PATTERNS:
            a_pos = bool(re.search(pos_pat, text_a, re.IGNORECASE))
            a_neg = bool(re.search(neg_pat, text_a, re.IGNORECASE))
            b_pos = bool(re.search(pos_pat, text_b, re.IGNORECASE))
            b_neg = bool(re.search(neg_pat, text_b, re.IGNORECASE))
            if (a_pos and b_neg) or (a_neg and b_pos):
                return True
        return False

    def _path_exists(self, rel_path: str) -> bool:
        return os.path.exists(os.path.join(os.getcwd(), rel_path))

