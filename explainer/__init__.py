"""
explainer — warstwa wyjaśnień LLM dla ProveNuance3.

Publiczne API:
    from explainer import LLMExplainer

Przykład użycia:
    from config import ExplainerConfig
    from explainer import LLMExplainer

    explainer = LLMExplainer(ExplainerConfig())
    explanation = explainer.explain(
        case_text=open("text_cases/TXT-003.txt").read(),
        facts=pipeline_result.facts,
        proof_run=proof_run,
        cluster_states=pipeline_result.cluster_states,
        entities=extraction_result.entities,
    )
    print(explanation)
"""
from .explainer import LLMExplainer

__all__ = ["LLMExplainer"]
