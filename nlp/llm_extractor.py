"""
LLMExtractor — ekstraktor faktów z polskiego tekstu oparty o Google Gemini.

Interfejs identyczny z TextExtractor:
    extractor = LLMExtractor(cluster_schemas, config)
    result = extractor.extract(text, source_id="case1")

Przepływ:
  1. Zbuduj system prompt z ontologii (predykaty + klastry z schematów)
  2. Wywołaj Gemini API (structured JSON output)
  3. Wykryj konflikty w surowej odpowiedzi (cluster + fact)
  4. Jeśli konflikty i pozostały retries → correction prompt → Gemini retry
  5. Parsuj final JSON → ExtractionResult
  6. (opcjonalnie) Uruchom SymbolicVerifier do walidacji strukturalnej
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from typing import Any

from nn.graph_builder import ClusterSchema
from runtime_env import get_required_env

from .llm_prompt import (
    build_correction_prompt,
    build_response_schema,
    build_system_prompt,
    parse_llm_response,
)
from .ontology_alignment import align_extraction_to_ontology
from .result import ExtractionResult


class LLMExtractor:
    """
    Ekstraktor faktów oparty o Google Gemini z pętlą zwrotną Symbolic Verifier.

    Parametry:
        cluster_schemas:   lista schematów klastrów (z DB / seed_ontology)
        config:            ExtractorConfig z config.py
        year:              rok domyślny dla dat bez roku (np. "1 lutego" → 2026-02-01)
    """

    def __init__(
        self,
        cluster_schemas: list[ClusterSchema],
        config: Any,  # ExtractorConfig — import lazy by uniknąć cyklicznych importów
        year: int = 2026,
        predicate_positions: dict[str, list[str]] | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "LLMExtractor wymaga pakietu google-genai. "
                "Zainstaluj: pip install google-genai"
            ) from exc

        api_key = get_required_env(config.api_key_env)
        self._client = genai.Client(api_key=api_key)
        if predicate_positions:
            self._predicate_positions = predicate_positions
        else:
            warnings.warn(
                "LLMExtractor: predicate_positions nie zostały przekazane — "
                "używam statycznego sv.schema.PREDICATE_POSITIONS. "
                "Przekaż predicate_positions z DBSession aby używać aktywnej ontologii.",
                stacklevel=2,
            )
            from sv.schema import PREDICATE_POSITIONS
            self._predicate_positions = dict(PREDICATE_POSITIONS)
        self._system_prompt = build_system_prompt(cluster_schemas, self._predicate_positions)
        self._response_schema = build_response_schema()
        self._schemas = cluster_schemas
        self._config = config
        self._year = year

        # SymbolicVerifier (opcjonalny) — lazy import by nie wymagać clingo gdy sv_verification=False
        self._sv = None
        if config.sv_verification:
            try:
                from sv import SymbolicVerifier
                self._sv = SymbolicVerifier(
                    cluster_schemas=cluster_schemas,
                    predicate_positions=self._predicate_positions,
                )
            except ImportError:
                pass  # clingo niedostępne — pomijamy SV walidację

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def extract(self, text: str, source_id: str = "text") -> ExtractionResult:
        """
        Ekstrahuje encje, fakty i stany klastrów z tekstu przy pomocy Gemini.

        Pętla zwrotna SV:
          - Gemini zwraca surowy JSON
          - Wykrywamy konflikty (cluster_name duplicate, fact contradictions)
          - Jeśli konflikty → correction_prompt → Gemini retry (max_retries razy)
          - Parsujemy final odpowiedź → ExtractionResult

        Zwraca ExtractionResult z:
          - entities:       encje explicit + implicit (CUST1, STORE1) + syntetyczne
          - facts:          reifikowane fakty (status=observed)
          - cluster_states: stany klastrów z clamp_source='text'
        """
        current_prompt: str = text
        best_result: ExtractionResult | None = None
        last_raw: dict[str, Any] | None = None

        for attempt in range(self._config.max_retries + 1):
            raw = self._call_gemini(current_prompt)
            conflicts = self._find_conflicts(raw)

            if not conflicts:
                # Brak konfliktów — parsujemy i zwracamy
                result = parse_llm_response(raw, source_id, self._year, self._schemas, text=text)
                result = self._align(result)
                if self._sv is not None:
                    result = self._sv_validate(result)
                return result

            # Zapamiętaj najlepszy dotychczasowy wynik
            if best_result is None:
                last_raw = raw

            if attempt < self._config.max_retries:
                # Zbuduj correction prompt i spróbuj ponownie
                current_prompt = build_correction_prompt(text, conflicts)
            else:
                # Wyczerpano retries — parsuj z odrzuconymi konfliktami
                raw_to_use = last_raw or raw
                result = parse_llm_response(raw_to_use, source_id, self._year, self._schemas, text=text)
                result = self._align(result)
                result = self._mark_conflicted(result, raw_to_use)
                if self._sv is not None:
                    result = self._sv_validate(result)
                return result

        # Nie powinniśmy tu dotrzeć — powyższa pętla zawsze zwraca
        raw_fallback = last_raw or {}
        return self._align(parse_llm_response(raw_fallback, source_id, self._year, self._schemas, text=text))

    # ------------------------------------------------------------------
    # Inspekcja promptu (przed wysłaniem)
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Zwraca system prompt wysyłany do Gemini (do podglądu / debugowania)."""
        return self._system_prompt

    def preview_request(self, text: str) -> dict[str, str]:
        """
        Zwraca słownik z pełną treścią żądania bez wywoływania API.

        Przykład:
            extractor = LLMExtractor(schemas, config)
            req = extractor.preview_request("Złożyłem zamówienie O100 1 marca.")
            print(req["system_prompt"])
            print(req["user_message"])
        """
        return {
            "system_prompt": self._system_prompt,
            "user_message": text,
            "model": self._config.gemini_model,
            "temperature": str(self._config.temperature),
            "response_mime_type": "application/json",
        }

    # ------------------------------------------------------------------
    # Gemini API
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        """Wywołaj Gemini i zwróć sparsowany JSON."""
        response = self._client.models.generate_content(
            model=self._config.gemini_model,
            contents=prompt,
            config={
                "system_instruction": self._system_prompt,
                "temperature": self._config.temperature,
                "response_mime_type": "application/json",
                "response_schema": self._response_schema,
            },
        )
        response_text = response.text
        if response_text is None:
            raise ValueError("Gemini returned an empty response body")
        return json.loads(response_text)

    # ------------------------------------------------------------------
    # Wykrywanie konfliktów w surowej odpowiedzi LLM
    # ------------------------------------------------------------------

    def _find_conflicts(self, raw: dict[str, Any]) -> list[str]:
        """
        Wykrywa sprzeczności w surowej odpowiedzi JSON z Gemini przed parsowaniem.

        Sprawdza:
        1. Cluster konflikty: ta sama (entity_id, cluster_name) z różnymi wartościami
        2. Fakt konflikty: ten sam (predicate, obligatoryjne_args) z różnymi opcjonalnymi args

        Zwraca listę opisów konfliktów po polsku, gotowych do wstawienia w correction prompt.
        """
        conflicts: list[str] = []

        # 1. Konflikty klastrów ─────────────────────────────────────────────
        # Grupuj po (entity_id, cluster_name) → zbierz unikalne wartości
        cluster_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for cs in raw.get("cluster_states", []):
            eid = str(cs.get("entity_id", "")).strip()
            cname = str(cs.get("cluster_name", "")).strip()
            value = str(cs.get("value", "")).strip().upper()
            if eid and cname and value:
                key = (eid, cname)
                if value not in cluster_groups[key]:
                    cluster_groups[key].append(value)

        for (eid, cname), values in cluster_groups.items():
            if len(values) > 1:
                vals_str = ", ".join(values)
                conflicts.append(
                    f"KLASTER {cname}: encja {eid} ma wiele wartości: {vals_str} "
                    f"— wybierz dokładnie jedną"
                )

        # 2. Konflikty faktów ───────────────────────────────────────────────
        # Grupuj po (predicate, wymagane_role) → zbierz opcjonalne role
        # Wymagane role = te bez DATE, COUPON, COMPLAINT, ACCOUNT, CHARGEBACK
        _optional_roles = frozenset({
            "DATE", "COUPON", "COMPLAINT", "ACCOUNT", "CHARGEBACK",
            "DELIVERY", "STATEMENT", "RETURN_SHIPMENT", "PAYMENT",
        })

        fact_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for f in raw.get("facts", []):
            predicate = str(f.get("predicate", "")).strip().upper()
            if not predicate:
                continue
            args = f.get("args", [])

            # Klucz = predicate + wymagane role (posortowane)
            key_parts = [predicate]
            for a in args:
                role = str(a.get("role", "")).strip().upper()
                if role not in _optional_roles:
                    val = str(a.get("entity_id") or a.get("literal_value") or "").strip()
                    if val:
                        key_parts.append(f"{role}:{val}")
            key_parts.sort()
            fact_key = tuple(key_parts)
            fact_groups[fact_key].append(f)

        for fact_key, instances in fact_groups.items():
            if len(instances) <= 1:
                continue
            # Sprzeczność = te same obligatoryjne args, ale różne opcjonalne
            # Opisz każdą instancję przez jej opcjonalne role
            optional_combos: list[str] = []
            for inst in instances:
                opt_parts = []
                for a in inst.get("args", []):
                    role = str(a.get("role", "")).strip().upper()
                    if role in _optional_roles:
                        val = str(a.get("entity_id") or a.get("literal_value") or "").strip()
                        if val:
                            opt_parts.append(f"{role}={val}")
                optional_combos.append("{" + ", ".join(opt_parts) + "}" if opt_parts else "{brak}")

            # Jeśli wszystkie combos są identyczne — to duplikaty, nie konflikty (OK)
            if len(set(optional_combos)) > 1:
                pred = fact_key[0]
                req_str = ", ".join(k for k in fact_key[1:])
                conflicts.append(
                    f"FAKT {pred}({req_str}): wiele wystąpień z różnymi opcjonalnymi "
                    f"argumentami: {'; '.join(optional_combos)} "
                    f"— zostaw tylko jedno lub pomiń duplikat"
                )

        return conflicts

    # ------------------------------------------------------------------
    # SV walidacja (po parsowaniu)
    # ------------------------------------------------------------------

    def _sv_validate(self, result: ExtractionResult) -> ExtractionResult:
        """
        Uruchamia SymbolicVerifier bez reguł na wyekstrahowanych faktach.
        Zwraca ExtractionResult ze zaktualizowanymi statusami faktów
        (observed → proved gdy SV potwierdzi; np. przy pustych regułach — bez zmian).
        """
        if self._sv is None:
            return result
        try:
            verify_result = self._sv.verify(
                facts=result.facts,
                rules=[],
                cluster_states=result.cluster_states,
            )
            return ExtractionResult(
                entities=result.entities,
                facts=verify_result.updated_facts,
                cluster_states=result.cluster_states,
                source_id=result.source_id,
            )
        except Exception:
            # SV błąd nie blokuje ekstrakcji — zwróć result bez zmian
            return result

    def _mark_conflicted(
        self, result: ExtractionResult, raw: dict[str, Any]
    ) -> ExtractionResult:
        """
        Po wyczerpaniu retries oznacza konflikty klastrów jako usunięte
        (zostawiamy tylko pierwszą wartość — już w parse_llm_response).
        Fakty pozostają bez zmian (parse_llm_response deduplikuje).
        Zwraca result bez modyfikacji (deduplication nastąpiła już w parserze).
        """
        return result

    def _align(self, result: ExtractionResult) -> ExtractionResult:
        return align_extraction_to_ontology(
            result,
            predicate_positions=self._predicate_positions,
            cluster_schemas=self._schemas,
            year=self._year,
        )
