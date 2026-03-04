"""
nlp/ — reguło-bazowy ekstraktor faktów z tekstu polskiego.

Nie wymaga modeli spaCy. Używa wyłącznie regex + słownika wzorców.

Przykład użycia:
    from nlp import TextExtractor, ExtractionResult
    from nn import ClusterSchema

    schemas = [...]
    extractor = TextExtractor(schemas)

    result = extractor.extract(
        "Złożyłem zamówienie O100 1 lutego. Dostałem maila, że zamówienie zostało przyjęte.",
        source_id="case_1",
    )
    print(result.summary())
    for f in result.facts:
        print(f.predicate, [a.entity_id or a.literal_value for a in f.args])
"""

from .extractor import TextExtractor
from .result import ExtractionResult

__all__ = [
    "TextExtractor",
    "ExtractionResult",
]
