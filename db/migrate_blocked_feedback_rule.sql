INSERT INTO rule_modules (name, description)
VALUES ('return', 'Return-related rules')
ON CONFLICT (name) DO UPDATE
SET description = COALESCE(rule_modules.description, EXCLUDED.description);

INSERT INTO rules (
    rule_id,
    module_id,
    language,
    head,
    body,
    clingo_text,
    stratum,
    learned,
    enabled,
    source_span_text
)
VALUES (
    'return.customer_bears_return_cost_before_return',
    (SELECT id FROM rule_modules WHERE name = 'return'),
    'horn_naf_stratified'::rule_language,
    $json$
    {
      "predicate": "customer_bears_return_cost",
      "args": [
        {"role": "CUSTOMER", "term": {"var": "C"}},
        {"role": "PRODUCT", "term": {"var": "P"}}
      ]
    }
    $json$::jsonb,
    $json$
    [
      {
        "literal_type": "pos",
        "predicate": "customer_can_withdraw",
        "args": [
          {"role": "CUSTOMER", "term": {"var": "C"}},
          {"role": "AGREEMENT", "term": {"var": "A"}}
        ]
      },
      {
        "literal_type": "pos",
        "predicate": "agreement_for_product",
        "args": [
          {"role": "AGREEMENT", "term": {"var": "A"}},
          {"role": "PRODUCT", "term": {"var": "P"}}
        ]
      },
      {
        "literal_type": "naf",
        "predicate": "product_returned",
        "args": [
          {"role": "PRODUCT", "term": {"var": "P"}},
          {"role": "CUSTOMER", "term": {"var": "C"}},
          {"role": "DATE", "term": {"var": "_"}}
        ]
      }
    ]
    $json$::jsonb,
    'customer_bears_return_cost(C,P) :- customer_can_withdraw(C,A), agreement_for_product(A,P), not product_returned(P,C,_).',
    1,
    FALSE,
    TRUE,
    'Manual NAF rule for blocked verifier feedback demo'
)
ON CONFLICT (rule_id) DO UPDATE SET
    module_id = EXCLUDED.module_id,
    language = EXCLUDED.language,
    head = EXCLUDED.head,
    body = EXCLUDED.body,
    clingo_text = EXCLUDED.clingo_text,
    stratum = EXCLUDED.stratum,
    learned = EXCLUDED.learned,
    enabled = TRUE,
    source_span_text = EXCLUDED.source_span_text,
    updated_at = now();
