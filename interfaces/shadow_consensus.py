from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ShadowConsensusEvaluation:
    consensus_id: str
    adapter_name: str
    operation: str
    comparison_profile: str
    live_provider: str | None
    live_status: str
    shadow_providers: list[str] = field(default_factory=list)
    usable_shadow_count: int = 0
    consensus_provider: str | None = None
    consensus_support: int = 0
    required_support: int = 1
    consensus_strength: float | None = None
    pairwise_min_agreement: float = 0.0
    live_agreement_score: float | None = None
    required_live_agreement: float = 0.0
    correctness_pass: bool = False
    reasons: list[str] = field(default_factory=list)
    request_summary: dict[str, Any] = field(default_factory=dict)
    live_summary: dict[str, Any] = field(default_factory=dict)
    consensus_summary: dict[str, Any] = field(default_factory=dict)
    member_scores: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        operation: str,
        comparison_profile: str,
        live_provider: str | None,
        live_status: str,
        shadow_providers: list[str],
        usable_shadow_count: int,
        consensus_provider: str | None,
        consensus_support: int,
        required_support: int,
        consensus_strength: float | None,
        pairwise_min_agreement: float,
        live_agreement_score: float | None,
        required_live_agreement: float,
        correctness_pass: bool,
        reasons: list[str] | None = None,
        request_summary: dict[str, Any] | None = None,
        live_summary: dict[str, Any] | None = None,
        consensus_summary: dict[str, Any] | None = None,
        member_scores: list[dict[str, Any]] | None = None,
    ) -> "ShadowConsensusEvaluation":
        return cls(
            consensus_id=f"shadowconsensus_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            operation=operation,
            comparison_profile=comparison_profile,
            live_provider=live_provider,
            live_status=live_status,
            shadow_providers=list(shadow_providers),
            usable_shadow_count=int(usable_shadow_count),
            consensus_provider=consensus_provider,
            consensus_support=int(consensus_support),
            required_support=int(required_support),
            consensus_strength=None if consensus_strength is None else round(float(consensus_strength), 4),
            pairwise_min_agreement=round(float(pairwise_min_agreement), 4),
            live_agreement_score=None if live_agreement_score is None else round(float(live_agreement_score), 4),
            required_live_agreement=round(float(required_live_agreement), 4),
            correctness_pass=bool(correctness_pass),
            reasons=list(reasons or []),
            request_summary=dict(request_summary or {}),
            live_summary=dict(live_summary or {}),
            consensus_summary=dict(consensus_summary or {}),
            member_scores=list(member_scores or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShadowConsensusScorer:
    """Task-aware consensus scorer for harder LLM and hybrid shadow comparisons."""

    def __init__(
        self,
        *,
        adapter_name: str,
        request_summary: dict[str, Any],
        operation: str,
        live_threshold_base: float,
        min_support: int,
        pairwise_min_agreement: float,
    ) -> None:
        self.adapter_name = adapter_name
        self.request_summary = dict(request_summary)
        self.operation = operation
        self.live_threshold_base = float(live_threshold_base)
        self.min_support = int(max(1, min_support))
        self.pairwise_min_agreement = float(pairwise_min_agreement)
        self.profile = self._infer_profile()
        self.required_live_agreement = self._profile_threshold(self.profile)

    def evaluate(
        self,
        *,
        live_provider: str | None,
        live_result: dict[str, Any],
        shadow_candidates: list[dict[str, Any]],
        live_summary: dict[str, Any],
    ) -> ShadowConsensusEvaluation:
        reasons: list[str] = []
        live_ok = live_result.get("status") == "ok"
        usable = [candidate for candidate in shadow_candidates if candidate.get("result", {}).get("status") == "ok"]
        if not live_ok:
            reasons.append(f"live_status:{live_result.get('status')}")
        if not usable:
            reasons.append("no_usable_shadow_candidates")
            return ShadowConsensusEvaluation.new(
                adapter_name=self.adapter_name,
                operation=self.operation,
                comparison_profile=self.profile,
                live_provider=live_provider,
                live_status=str(live_result.get("status", "unknown")),
                shadow_providers=[str(candidate.get("provider")) for candidate in shadow_candidates],
                usable_shadow_count=0,
                consensus_provider=None,
                consensus_support=0,
                required_support=self.min_support,
                consensus_strength=None,
                pairwise_min_agreement=self.pairwise_min_agreement,
                live_agreement_score=None,
                required_live_agreement=self.required_live_agreement,
                correctness_pass=bool(live_ok),
                reasons=reasons,
                request_summary=self.request_summary,
                live_summary=live_summary,
                consensus_summary={},
                member_scores=[],
            )

        matrix = self._pairwise_matrix(usable)
        member_scores = self._member_scores(usable, matrix)
        winner_index = self._select_winner_index(member_scores)
        winner = usable[winner_index]
        consensus_support = int(member_scores[winner_index]["support"])
        consensus_strength = float(member_scores[winner_index]["peer_average"]) if usable else None
        consensus_summary = self._summarize_result(winner.get("result", {}))
        live_agreement_score = self.score_pair(live_result, winner.get("result", {})) if live_ok else None

        correctness_pass = bool(live_ok)
        if consensus_support < self.min_support:
            correctness_pass = False
            reasons.append(f"consensus_support_below_threshold:{consensus_support}<{self.min_support}")
        if live_agreement_score is None:
            correctness_pass = False
            reasons.append("live_agreement_unavailable")
        elif live_agreement_score < self.required_live_agreement:
            correctness_pass = False
            reasons.append(
                f"consensus_disagreement:{live_agreement_score:.3f}<{self.required_live_agreement:.3f}"
            )

        return ShadowConsensusEvaluation.new(
            adapter_name=self.adapter_name,
            operation=self.operation,
            comparison_profile=self.profile,
            live_provider=live_provider,
            live_status=str(live_result.get("status", "unknown")),
            shadow_providers=[str(candidate.get("provider")) for candidate in shadow_candidates],
            usable_shadow_count=len(usable),
            consensus_provider=str(winner.get("provider")),
            consensus_support=consensus_support,
            required_support=self.min_support,
            consensus_strength=consensus_strength,
            pairwise_min_agreement=self.pairwise_min_agreement,
            live_agreement_score=live_agreement_score,
            required_live_agreement=self.required_live_agreement,
            correctness_pass=correctness_pass,
            reasons=reasons,
            request_summary=self.request_summary,
            live_summary=live_summary,
            consensus_summary=consensus_summary,
            member_scores=member_scores,
        )

    def score_pair(self, left_result: dict[str, Any], right_result: dict[str, Any]) -> float | None:
        if left_result.get("status") != "ok" or right_result.get("status") != "ok":
            return None
        if self.adapter_name == "quantum_solver":
            return self._score_quantum_pair(left_result, right_result)
        left_text = str(left_result.get("text", ""))
        right_text = str(right_result.get("text", ""))
        if self.profile == "classification":
            left_label = self._extract_label(left_text)
            right_label = self._extract_label(right_text)
            if left_label and right_label:
                return 1.0 if left_label == right_label else 0.0
            return self._freeform_similarity(left_text, right_text)
        if self.profile == "json":
            left_json = self._try_parse_json(left_text)
            right_json = self._try_parse_json(right_text)
            if left_json is not None and right_json is not None:
                return self._json_similarity(left_json, right_json)
            return self._freeform_similarity(left_text, right_text)
        if self.profile == "qa":
            return self._qa_similarity(left_text, right_text)
        if self.profile == "summary":
            return self._summary_similarity(left_text, right_text)
        return self._freeform_similarity(left_text, right_text)

    def _infer_profile(self) -> str:
        task = str(self.request_summary.get("task") or "").strip().lower()
        operation = self.operation.lower()
        if self.adapter_name == "quantum_solver":
            return "quantum"
        if task in {"summarize", "summary"}:
            return "summary"
        if task in {"classify", "classification", "classify_risk", "risk_classification"}:
            return "classification"
        if task in {"structured_answer", "structured_extract", "json_extract", "json"}:
            return "json"
        if task in {"qa", "qa_short", "answer", "answer_short"}:
            return "qa"
        if "classif" in operation:
            return "classification"
        if "extract" in operation or "json" in operation:
            return "json"
        if "answer" in operation or "qa" in operation:
            return "qa"
        return "freeform"

    def _profile_threshold(self, profile: str) -> float:
        base = self.live_threshold_base
        if profile == "classification":
            return max(base, 0.95)
        if profile == "json":
            return max(base, 0.85)
        if profile == "qa":
            return max(base, 0.8)
        if profile == "quantum":
            return max(base, 0.99)
        return max(base, 0.35)

    def _pairwise_matrix(self, usable: list[dict[str, Any]]) -> list[list[float | None]]:
        matrix: list[list[float | None]] = [[None for _ in usable] for _ in usable]
        for left_index, left in enumerate(usable):
            for right_index, right in enumerate(usable):
                if left_index == right_index:
                    matrix[left_index][right_index] = 1.0
                elif matrix[left_index][right_index] is None:
                    score = self.score_pair(left.get("result", {}), right.get("result", {}))
                    matrix[left_index][right_index] = score
                    matrix[right_index][left_index] = score
        return matrix

    def _member_scores(self, usable: list[dict[str, Any]], matrix: list[list[float | None]]) -> list[dict[str, Any]]:
        member_scores: list[dict[str, Any]] = []
        for index, candidate in enumerate(usable):
            peer_scores = [float(score) for peer_index, score in enumerate(matrix[index]) if peer_index != index and score is not None]
            peer_average = (sum(peer_scores) / len(peer_scores)) if peer_scores else 1.0
            support = 1 + sum(1 for score in peer_scores if score >= self.pairwise_min_agreement)
            member_scores.append(
                {
                    "provider": str(candidate.get("provider")),
                    "peer_average": round(float(peer_average), 4),
                    "support": int(support),
                    "latency_ms": None if candidate.get("latency_ms") is None else round(float(candidate.get("latency_ms")), 3),
                }
            )
        return member_scores

    def _select_winner_index(self, member_scores: list[dict[str, Any]]) -> int:
        best_index = 0
        best_tuple = (-1.0, -1, float("inf"), "")
        for index, candidate in enumerate(member_scores):
            latency = float(candidate.get("latency_ms") or 0.0)
            key = (
                float(candidate.get("peer_average", 0.0)),
                int(candidate.get("support", 0)),
                -latency,
                str(candidate.get("provider", "")),
            )
            if key > best_tuple:
                best_tuple = key
                best_index = index
        return best_index

    def _score_quantum_pair(self, left_result: dict[str, Any], right_result: dict[str, Any]) -> float | None:
        if "factors" in left_result and "factors" in right_result:
            return 1.0 if sorted(left_result.get("factors", [])) == sorted(right_result.get("factors", [])) else 0.0
        if "best_value" in left_result and "best_value" in right_result:
            left = float(left_result.get("best_value"))
            right = float(right_result.get("best_value"))
            if left == right:
                return 1.0
            return max(0.0, 1.0 - abs(left - right))
        return None

    def _summary_similarity(self, left: str, right: str) -> float:
        token_overlap = self._token_jaccard(left, right)
        sentence_overlap = self._sentence_overlap(left, right)
        number_overlap = self._number_similarity(left, right)
        length_similarity = self._length_similarity(left, right)
        return min(1.0, max(0.0, (0.45 * token_overlap) + (0.30 * sentence_overlap) + (0.15 * number_overlap) + (0.10 * length_similarity)))

    def _qa_similarity(self, left: str, right: str) -> float:
        normalized_left = self._normalize_short_answer(left)
        normalized_right = self._normalize_short_answer(right)
        if normalized_left and normalized_right and normalized_left == normalized_right:
            return 1.0
        if normalized_left and normalized_right and (normalized_left in normalized_right or normalized_right in normalized_left):
            return 0.9
        return max(self._token_jaccard(left, right), self._length_similarity(left, right) * 0.5)

    def _freeform_similarity(self, left: str, right: str) -> float:
        token_overlap = self._token_jaccard(left, right)
        length_similarity = self._length_similarity(left, right)
        return min(1.0, max(0.0, (0.75 * token_overlap) + (0.25 * length_similarity)))

    def _json_similarity(self, left: Any, right: Any) -> float:
        if isinstance(left, dict) and isinstance(right, dict):
            keys = sorted(set(left.keys()) | set(right.keys()))
            if not keys:
                return 1.0
            scores = [self._json_similarity(left.get(key), right.get(key)) for key in keys]
            return sum(scores) / len(scores)
        if isinstance(left, list) and isinstance(right, list):
            if not left and not right:
                return 1.0
            if not left or not right:
                return 0.0
            comparable = min(len(left), len(right))
            scores = [self._json_similarity(left[index], right[index]) for index in range(comparable)]
            length_penalty = comparable / max(len(left), len(right), 1)
            return (sum(scores) / len(scores)) * length_penalty if scores else 0.0
        if left is None and right is None:
            return 1.0
        if left is None or right is None:
            return 0.0
        if isinstance(left, (int, float)) or isinstance(right, (int, float)):
            try:
                left_value = float(left)
                right_value = float(right)
            except (TypeError, ValueError):
                return 1.0 if str(left).strip().lower() == str(right).strip().lower() else 0.0
            if left_value == right_value:
                return 1.0
            return max(0.0, 1.0 - (abs(left_value - right_value) / max(abs(left_value), abs(right_value), 1.0)))
        return 1.0 if str(left).strip().lower() == str(right).strip().lower() else self._token_jaccard(str(left), str(right))

    def _try_parse_json(self, text: str) -> Any | None:
        candidate = str(text).strip()
        if not candidate:
            return None
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
            candidate = re.sub(r"```$", "", candidate).strip()
        if not candidate.startswith(("{", "[")):
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _extract_label(self, text: str) -> str | None:
        parsed = self._try_parse_json(text)
        if isinstance(parsed, dict):
            for key in ("label", "class", "decision", "category"):
                if key in parsed and parsed[key] is not None:
                    return self._normalize_short_answer(str(parsed[key]))
        pattern = re.compile(r"(?:label|class|decision|category)\s*[:=]\s*[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)
        match = pattern.search(str(text))
        if match:
            return self._normalize_short_answer(match.group(1))
        cleaned = self._normalize_short_answer(text)
        if cleaned and " " not in cleaned:
            return cleaned
        return None

    def _normalize_short_answer(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Za-zА-Яа-я0-9_\-\s]", " ", str(text).lower())
        cleaned = " ".join(cleaned.split())
        prefixes = ("answer ", "label ", "class ", "decision ")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        return cleaned

    def _token_jaccard(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", str(left).lower()))
        right_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_]+", str(right).lower()))
        if not left_tokens and not right_tokens:
            return 1.0
        if not left_tokens or not right_tokens:
            return 0.0
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0

    def _sentence_overlap(self, left: str, right: str) -> float:
        left_sentences = {self._normalize_sentence(item) for item in re.split(r"[.!?;\n]+", str(left)) if self._normalize_sentence(item)}
        right_sentences = {self._normalize_sentence(item) for item in re.split(r"[.!?;\n]+", str(right)) if self._normalize_sentence(item)}
        if not left_sentences and not right_sentences:
            return 1.0
        if not left_sentences or not right_sentences:
            return 0.0
        union = left_sentences | right_sentences
        return len(left_sentences & right_sentences) / len(union) if union else 0.0

    def _normalize_sentence(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Za-zА-Яа-я0-9_\s]", " ", str(text).lower())
        return " ".join(cleaned.split())

    def _number_similarity(self, left: str, right: str) -> float:
        left_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", str(left)))
        right_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", str(right)))
        if not left_numbers and not right_numbers:
            return 1.0
        if not left_numbers or not right_numbers:
            return 0.0
        union = left_numbers | right_numbers
        return len(left_numbers & right_numbers) / len(union) if union else 0.0

    def _length_similarity(self, left: str, right: str) -> float:
        left_length = len(str(left).strip())
        right_length = len(str(right).strip())
        if left_length == 0 and right_length == 0:
            return 1.0
        return max(0.0, 1.0 - (abs(left_length - right_length) / max(left_length, right_length, 1)))

    def _summarize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {"status": result.get("status"), "reason": result.get("reason")}
        if "text" in result:
            text = str(result.get("text", ""))
            summary["text_preview"] = text[:120]
            summary["text_length"] = len(text)
            label = self._extract_label(text)
            if label:
                summary["label"] = label
            parsed = self._try_parse_json(text)
            if isinstance(parsed, dict):
                summary["json_keys"] = sorted(parsed.keys())[:10]
        if "factors" in result:
            summary["factors"] = list(result.get("factors", []))[:4]
        if "best_value" in result:
            summary["best_value"] = result.get("best_value")
            summary["best_index"] = result.get("best_index")
        return summary
