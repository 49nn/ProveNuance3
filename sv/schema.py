"""
Konfiguracja projektu: kolejność ról pozycyjnych predykatów (z predicate_roles w DB)
i role nazw klastrów (z reguł w seed_ontology.sql).

Służy do:
  1. Konwersji Fact.args (role-based) → Clingo LP string (positional).
  2. Konwersji clingo.Symbol → GroundAtom (positional → role-based).
"""

# ---------------------------------------------------------------------------
# Kolejność ról pozycyjnych dla 19 predykatów n-arnych
# Źródło: predicate_roles w db/seed_ontology.sql (position 0, 1, 2, ...)
# ---------------------------------------------------------------------------

PREDICATE_POSITIONS: dict[str, list[str]] = {
    "order_placed":                   ["CUSTOMER", "ORDER",      "DATE"],
    "order_accepted":                 ["STORE",    "ORDER",      "DATE"],
    "payment_selected":               ["CUSTOMER", "ORDER",      "METHOD"],
    "payment_made":                   ["CUSTOMER", "ORDER",      "PAYMENT", "DATE", "AMOUNT"],
    "payment_due_by":                 ["ORDER",    "DATE"],
    "order_cancelled":                ["STORE",    "ORDER",      "DATE",    "REASON"],
    "delivered":                      ["ORDER",    "DELIVERY",   "DATE"],
    "digital_access_granted":         ["ORDER",    "DATE"],
    "download_started":               ["ORDER",    "DATE"],
    "withdrawal_statement_submitted": ["CUSTOMER", "ORDER",      "STATEMENT", "DATE"],
    "returned":                       ["ORDER",    "RETURN_SHIPMENT", "DATE"],
    "return_proof_provided":          ["ORDER",    "DATE"],
    "refund_issued":                  ["ORDER",    "PAYMENT",    "DATE",   "AMOUNT"],
    "complaint_submitted":            ["CUSTOMER", "ORDER",      "COMPLAINT", "DATE"],
    "complaint_response_sent":        ["STORE",    "COMPLAINT",  "DATE"],
    "coupon_applied":                 ["CUSTOMER", "ORDER",      "COUPON", "DATE"],
    "account_blocked":                ["STORE",    "ACCOUNT",    "DATE",   "REASON"],
    "chargeback_opened":              ["CUSTOMER", "ORDER",      "CHARGEBACK", "DATE"],
    "chargeback_resolved":            ["CHARGEBACK", "DATE",     "RESULT"],
}

# ---------------------------------------------------------------------------
# Role (entity_role, value_role) dla 12 klastrów unarnych
# Kolejność: [entity_role, value_role] odpowiada kolejności w Clingo LP
# Źródło: jak klastry pojawiają się w ciałach reguł w seed_ontology.sql
# ---------------------------------------------------------------------------

DEFAULT_CLUSTER_ROLES: dict[str, tuple[str, str]] = {
    "customer_type":         ("CUSTOMER", "TYPE"),
    "order_status":          ("ORDER",    "VALUE"),
    "payment_method":        ("ORDER",    "METHOD"),
    "product_type":          ("PRODUCT",  "TYPE"),
    "defective":             ("ORDER",    "VALUE"),
    "store_pays_return":     ("ORDER",    "VALUE"),
    "digital_consent":       ("ORDER",    "VALUE"),
    "download_started_flag": ("ORDER",    "VALUE"),
    "coupon_stackable":      ("COUPON",   "VALUE"),
    "account_status":        ("ACCOUNT",  "VALUE"),
    "chargeback_status":     ("ORDER",    "VALUE"),
    "password_shared":       ("ACCOUNT",  "VALUE"),
}
