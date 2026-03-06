"""
LLMExplainer — wyjaśnianie wyników pipeline'u w języku naturalnym (Gemini).

Interfejs:
    explainer = LLMExplainer(config)
    explanation = explainer.explain(
        case_text="...",
        facts=result.facts,
        proof_run=proof_run,          # opcjonalnie — proweniencja symboliczna
        cluster_states=result.cluster_states,
        entities=result.entities,     # opcjonalnie — do mapowania ID -> nazwy
        neural_trace=trace_dict,      # opcjonalnie — proweniencja neuronalna
                                      #   dict[target_fact_id, list[NeuralTraceItem]]
    )
    print(explanation)  # naturalny tekst po polsku (lub angielsku)

Podgląd bez wywołania API:
    req = explainer.preview_request(case_text, facts, proof_run, cluster_states, entities)
    print(req["system_prompt"])
    print(req["user_message"])
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from data_model.entity import Entity
from data_model.fact import Fact, FactStatus, NeuralTraceItem
from runtime_env import get_required_env

from .prompt import build_system_prompt, build_user_message

if TYPE_CHECKING:
    from nn.graph_builder import ClusterStateRow
    from sv.proof import ProofRun


class LLMExplainer:
    """
    Wyjaśnia wyniki pipeline'u ProveNuance3 w języku naturalnym używając Google Gemini.

    Parametry:
        config: ExplainerConfig z config.py
    """

    def __init__(self, config: Any) -> None:  # Any = ExplainerConfig (lazy by uniknąć cykli)
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "LLMExplainer wymaga pakietu google-genai. "
                "Zainstaluj: pip install google-genai"
            ) from exc

        api_key = get_required_env(config.api_key_env)
        self._client = genai.Client(api_key=api_key)
        self._system_prompt = build_system_prompt(config.language)
        self._config = config

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def explain(
        self,
        case_text: str,
        facts: list[Fact],
        proof_run: "ProofRun | None" = None,
        cluster_states: "list[ClusterStateRow] | None" = None,
        entities: list[Entity] | None = None,
        neural_trace: dict[str, list[NeuralTraceItem]] | None = None,
    ) -> str:
        """
        Generuje wyjaśnienie sprawy w języku naturalnym.

        Parametry:
            case_text:      oryginalny tekst sprawy
            facts:          fakty z PipelineResult / ExtractionResult
            proof_run:      dowód logiczny z build_proof_run() (proweniencja symboliczna)
            cluster_states: stany klastrów (np. customer_type=BUSINESS)
            entities:       lista encji do mapowania ID -> canonical_name
            neural_trace:   dict[target_fact_id, list[NeuralTraceItem]] (proweniencja NN)
                            Budowany przez: {fid: tracer.finalize(fid) for fid in fact_ids}

        Zwraca:
            str — odpowiedź Gemini w naturalnym języku
        """
        entity_map = _build_entity_map(entities)
        filtered_facts = _filter_facts(facts, self._config.max_facts)
        user_msg = build_user_message(
            case_text=case_text,
            facts=filtered_facts,
            proof_run=proof_run,
            cluster_states=cluster_states,
            entity_map=entity_map,
            grounded=self._config.grounded,
            neural_trace=neural_trace,
        )
        response = self._client.models.generate_content(
            model=self._config.gemini_model,
            contents=user_msg,
            config={
                "system_instruction": self._system_prompt,
                "temperature": self._config.temperature,
                "response_mime_type": "text/plain",
            },
        )
        return response.text or ""

    def preview_request(
        self,
        case_text: str,
        facts: list[Fact],
        proof_run: "ProofRun | None" = None,
        cluster_states: "list[ClusterStateRow] | None" = None,
        entities: list[Entity] | None = None,
        neural_trace: dict[str, list[NeuralTraceItem]] | None = None,
    ) -> dict[str, str]:
        """
        Zwraca słownik z pełną treścią żądania bez wywoływania API.

        Przykład:
            req = explainer.preview_request(case_text, facts, proof_run, cluster_states)
            print(req["system_prompt"])
            print(req["user_message"])
        """
        entity_map = _build_entity_map(entities)
        filtered_facts = _filter_facts(facts, self._config.max_facts)
        user_msg = build_user_message(
            case_text=case_text,
            facts=filtered_facts,
            proof_run=proof_run,
            cluster_states=cluster_states,
            entity_map=entity_map,
            grounded=self._config.grounded,
            neural_trace=neural_trace,
        )
        return {
            "system_prompt": self._system_prompt,
            "user_message": user_msg,
            "model": self._config.gemini_model,
            "temperature": str(self._config.temperature),
            "language": self._config.language,
        }


# ------------------------------------------------------------------
# Pomocnicze
# ------------------------------------------------------------------

def _build_entity_map(entities: list[Entity] | None) -> dict[str, str]:
    """Buduje słownik entity_id → canonical_name."""
    if not entities:
        return {}
    return {e.entity_id: e.canonical_name for e in entities}


def _filter_facts(facts: list[Fact], max_facts: int) -> list[Fact]:
    """
    Filtruje i sortuje fakty do promptu:
      - Tylko status proved lub observed
      - proved najpierw
      - Limit max_facts
    """
    proved = [f for f in facts if f.status == FactStatus.proved]
    observed = [f for f in facts if f.status == FactStatus.observed]
    combined = proved + observed
    return combined[:max_facts]
