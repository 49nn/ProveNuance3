"""
Wzorce regułowe dla polskiego ekstractora faktów.

Zawiera:
  PL_MONTHS      — mapa nazw miesięcy (PL) → numer
  DATE_RE        — regex dat: "10 marca", "1 lutego"
  ENTITY_ID_PATTERNS — typy encji + wzorce ID (O[0-9]+, A[0-9]+, CB[0-9]+, C[0-9]{2,})
  COUPON_CONTEXT_RE  — kody kuponów w kontekście słów "kupon"/"kod"
  CLUSTER_RULES  — pary (wzorzec tekstu → cluster_name + wartość domeny)
  FACT_RULES     — pary (wzorzec tekstu → predykat + role)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Miesiące polskie
# ---------------------------------------------------------------------------

PL_MONTHS: dict[str, int] = {
    "stycznia": 1,  "styczeń": 1,   "styczen": 1,
    "lutego": 2,    "luty": 2,
    "marca": 3,     "marzec": 3,
    "kwietnia": 4,  "kwiecień": 4,  "kwiecien": 4,
    "maja": 5,      "maj": 5,
    "czerwca": 6,   "czerwiec": 6,
    "lipca": 7,     "lipiec": 7,
    "sierpnia": 8,  "sierpień": 8,  "sierpien": 8,
    "września": 9,  "wrzesień": 9,  "wrzesien": 9,
    "października": 10, "październik": 10, "pazdziernika": 10,
    "listopada": 11, "listopad": 11,
    "grudnia": 12,  "grudzień": 12, "grudzien": 12,
}

DATE_RE = re.compile(
    r"\b(\d{1,2})\s+("
    + "|".join(sorted(PL_MONTHS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Wzorce identyfikatorów encji
# Kolejność ma znaczenie: CHARGEBACK (CB\d+) przed COMPLAINT (C\d{2,}).
# ---------------------------------------------------------------------------

ENTITY_ID_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("CHARGEBACK", re.compile(r"\bCB\d+\b")),
    ("COMPLAINT",  re.compile(r"\bC\d{2,}\b")),   # C900 itp.
    ("ACCOUNT",    re.compile(r"\bA\d+\b")),
    ("ORDER",      re.compile(r"\bO\d+\b")),
    ("PRODUCT",    re.compile(r"\bP\d+\b")),
]

# Kody kuponów w bezpośrednim kontekście słów "kupon"/"kod"
COUPON_CONTEXT_RE = re.compile(
    r"(?:kupon(?:ów|y|em|ie|a)?|kod(?:u|em|y|ów)?)\s*:?\s*"
    r"([A-Z][A-Z0-9]*(?:\s+i\s+[A-Z][A-Z0-9]*)*)",
    re.IGNORECASE,
)

# Wyszukiwanie znanych kodów kuponów gdziekolwiek w tekście
# (używane do znalezienia wszystkich pozycji kodu po jego identyfikacji)
COUPON_CODE_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,})\b")

# Słowa uppercase, które NIE są kuponami
NON_COUPON_WORDS: frozenset[str] = frozenset({
    "BLIK", "COD", "YES", "NO", "CARD", "CUSTOM", "DIGITAL", "PHYSICAL",
    "TRANSFER", "CONSUMER", "BUSINESS", "ACTIVE", "BLOCKED", "OPEN", "NONE",
    "PLACED", "ACCEPTED", "PAID", "CANCELLED", "DELIVERED", "DISPUTED",
    "RESOLVED", "WON", "FRAUD", "NORMAL", "STORE", "CUST", "DEL",
    "STMT", "RET", "PAY", "PROD", "NAF",
})

# ---------------------------------------------------------------------------
# Klastry: reguła (wzorzec → cluster_name + wartość + typ encji)
# ---------------------------------------------------------------------------


@dataclass
class ClusterRule:
    pattern: re.Pattern
    cluster_name: str   # musi zgadzać się z nazwą schematu klastra
    value: str          # wartość domeny (np. "CONSUMER", "TRANSFER")
    entity_type: str    # typ encji właściciela klastra


CLUSTER_RULES: list[ClusterRule] = [
    # customer_type
    ClusterRule(
        re.compile(r"jestem\s+konsumentem|jako\s+konsument|osoba\s+fizyczna|"
                   r"jestem\s+(?:zwykłym\s+)?(?:klientem|konsumentem)", re.I),
        "customer_type", "CONSUMER", "CUSTOMER",
    ),
    ClusterRule(
        re.compile(r"jako\s+przedsiębiorca|jestem\s+przedsiębiorcą|"
                   r"firma|działalność\s+gospodarcza", re.I),
        "customer_type", "BUSINESS", "CUSTOMER",
    ),
    # payment_method
    ClusterRule(
        re.compile(r"\bprzelewem?\b|przelew\s+bankowy|wybrałem\s+przelew", re.I),
        "payment_method", "TRANSFER", "ORDER",
    ),
    ClusterRule(
        re.compile(r"\bBLIK\b"),
        "payment_method", "BLIK", "ORDER",
    ),
    ClusterRule(
        re.compile(r"przy\s+odbiorze|płatno(?:ść|sc)\s+przy\s+odbiorze|\bCOD\b"),
        "payment_method", "COD", "ORDER",
    ),
    ClusterRule(
        re.compile(r"\bkart[ąa](?:\s+kredytow[ąa]|\s+debetow[ąa])?\b|"
                   r"płatno(?:ść|sc)\s+kart[ąa]", re.I),
        "payment_method", "CARD", "ORDER",
    ),
    # product_type
    ClusterRule(
        re.compile(r"produkt\s+cyfrowy|treść\s+cyfrow[ąa]|link\s+do\s+pobrania|"
                   r"do\s+pobrania|treści\s+cyfrowej", re.I),
        "product_type", "DIGITAL", "PRODUCT",
    ),
    ClusterRule(
        re.compile(r"na\s+(?:zamówienie|wymiar|miarę|indywidualn)|"
                   r"meble\s+na|niestandardow", re.I),
        "product_type", "CUSTOM", "PRODUCT",
    ),
    ClusterRule(
        re.compile(r"produkt\s+fizyczny", re.I),
        "product_type", "PHYSICAL", "PRODUCT",
    ),
    # defective
    ClusterRule(
        re.compile(r"wadliwy|wada\s+(?:towaru|produktu)|towar\s+jest\s+wadliwy|"
                   r"uszkodzon(?:y|ego|emu)", re.I),
        "defective", "YES", "ORDER",
    ),
    # store_pays_return
    ClusterRule(
        re.compile(r"sklep\s+(?:zgodził\s+się|pokryje|zapłaci)\s+"
                   r"(?:koszt\s+)?(?:zwrotu|odesłania)|"
                   r"pokryć\s+koszt\s+(?:zwrotu|odesłania)", re.I),
        "store_pays_return", "YES", "ORDER",
    ),
    # digital_consent
    ClusterRule(
        re.compile(r"za\s+wyraźną\s+zgodą|wyraźn[ąa]\s+zgod[ęą]|"
                   r"zgodziłem\s+się\s+na\s+(?:rozpoczęcie|dostarczenie)|"
                   r"wyraziłem\s+zgodę", re.I),
        "digital_consent", "YES", "ORDER",
    ),
    # download_started_flag
    ClusterRule(
        re.compile(r"zacząłem\s+pobieranie|rozpoczął\s+pobieranie|"
                   r"pobranie\s+(?:zostało\s+)?(?:rozpoczęte|zaczęte)|"
                   r"zacząłem\s+korzystać", re.I),
        "download_started_flag", "YES", "ORDER",
    ),
    # account_status
    ClusterRule(
        re.compile(r"konto\s+(?:jest\s+)?zablokowane|"
                   r"zablokował\s+(?:moje\s+)?konto|"
                   r"konto\s+zostało\s+zablokowane", re.I),
        "account_status", "BLOCKED", "ACCOUNT",
    ),
    # chargeback_status
    ClusterRule(
        re.compile(r"bank\s+uznał\s+(?:chargeback\s+)?na\s+(?:moją\s+)?korzyść|"
                   r"wygrał\s+(?:chargeback\s+)?klient", re.I),
        "chargeback_status", "RESOLVED_CUSTOMER", "ORDER",
    ),
    ClusterRule(
        re.compile(r"bank\s+uznał\s+(?:reklamację\s+)?sklepu|wygrał\s+sklep|"
                   r"przegrał\s+klient|chargeback\s+przegrany\s+przez\s+klienta", re.I),
        "chargeback_status", "RESOLVED_STORE", "ORDER",
    ),
    ClusterRule(
        re.compile(r"otworzyłem\s+chargeback|złożyłem\s+chargeback|"
                   r"zgłosiłem\s+chargeback", re.I),
        "chargeback_status", "OPEN", "ORDER",
    ),
    # coupon_stackable
    ClusterRule(
        re.compile(r"nie\s+łączy\s+się\s+z\s+innymi|nie\s+łączą\s+się|"
                   r"nie\s+można\s+łączyć|nie\s+łączy\s+się", re.I),
        "coupon_stackable", "NO", "COUPON",
    ),
    # password_shared
    ClusterRule(
        re.compile(r"udostępniłem\s+hasło|podałem\s+hasło|"
                   r"podzieliłem\s+się\s+hasłem", re.I),
        "password_shared", "YES", "ACCOUNT",
    ),
]

# ---------------------------------------------------------------------------
# Fakty: reguła (wzorzec → predykat + specyfikacja ról)
#
# Specyfikatory ról (role_sources):
#   "IMPLICIT_CUSTOMER"     → entity_id = CUST1
#   "IMPLICIT_STORE"        → entity_id = STORE1
#   "ORDER"                 → najbliższy ORDER entity_id
#   "DATE"                  → najbliższy date entity_id (D_YYYY-MM-DD)
#   "ACCOUNT"               → najbliższy ACCOUNT entity_id
#   "CHARGEBACK"            → najbliższy CHARGEBACK entity_id
#   "COMPLAINT"             → najbliższy COMPLAINT entity_id
#   "COUPON"                → najbliższy COUPON entity_id
#   "AUTO_DELIVERY_ORDER"   → "DEL_" + najbliższy ORDER
#   "AUTO_STATEMENT_ORDER"  → "STMT_" + najbliższy ORDER
#   "AUTO_RETURN_ORDER"     → "RET_" + najbliższy ORDER
#   "AUTO_PAYMENT_ORDER"    → "PAY_" + najbliższy ORDER
#   "literal:{VALUE}"       → literal_value = VALUE
# ---------------------------------------------------------------------------


@dataclass
class FactRule:
    predicate: str
    trigger: re.Pattern
    role_sources: list[tuple[str, str]]   # [(role_name, source_spec)]


FACT_RULES: list[FactRule] = [
    # ── Zamówienia ───────────────────────────────────────────────────────────
    FactRule(
        "ORDER_PLACED",
        re.compile(
            r"złożył(?:em|a|eś)?\s+zamówienie|"
            r"zamówienie\s+(?:\w+\s+)?złożył(?:em|a|eś)?|"
            r"zamówienie\s+(?:zostało\s+)?złożon",
            re.I,
        ),
        [("CUSTOMER", "IMPLICIT_CUSTOMER"), ("ORDER", "ORDER"), ("DATE", "DATE")],
    ),
    FactRule(
        "ORDER_ACCEPTED",
        re.compile(
            r"zamówienie\s+zostało\s+przyjęte|potwierdzono?\s+przyjęcie|"
            r"dostał(?:em|am)\s+mail(?:a)?\s+(?:od\s+sklepu\s+)?(?:\w+\s+)?(?:że|o\s+tym)",
            re.I,
        ),
        [("STORE", "IMPLICIT_STORE"), ("ORDER", "ORDER"), ("DATE", "DATE")],
    ),
    FactRule(
        "ORDER_CANCELLED",
        re.compile(
            r"sklep\s+anulował\s+zamówienie|zamówienie\s+(?:zostało\s+)?anulowane",
            re.I,
        ),
        [
            ("STORE", "IMPLICIT_STORE"), ("ORDER", "ORDER"),
            ("DATE", "DATE"), ("REASON", "literal:NONPAYMENT"),
        ],
    ),
    # ── Dostawa ──────────────────────────────────────────────────────────────
    FactRule(
        "DELIVERED",
        re.compile(
            r"doręczon(?:o|e)|dostarczono\s+zamówienie|"
            r"zamówienie\s+(?:zostało\s+)?doręczon|"
            r"odebrał(?:em|am)\s+(?:zamówienie|towar)",
            re.I,
        ),
        [("ORDER", "ORDER"), ("DELIVERY", "AUTO_DELIVERY_ORDER"), ("DATE", "DATE")],
    ),
    # ── Odstąpienie ──────────────────────────────────────────────────────────
    FactRule(
        "WITHDRAWAL_STATEMENT_SUBMITTED",
        re.compile(
            r"(?:złożył(?:em|am)?|wysłał(?:em|am)?|zgłosił(?:em|am)?)\s+"
            r"oświadczenie\s+(?:o\s+odstąpieniu|odstąpienia)|"
            r"oświadczenie\s+o\s+odstąpieniu\s+(?:od\s+umowy\s+)?(?:wysłał(?:em)?|złożył(?:em)?)|"
            r"odstąpił(?:em|am)?\s+od\s+umowy",
            re.I,
        ),
        [
            ("CUSTOMER", "IMPLICIT_CUSTOMER"), ("ORDER", "ORDER"),
            ("STATEMENT", "AUTO_STATEMENT_ORDER"), ("DATE", "DATE"),
        ],
    ),
    # ── Zwrot ────────────────────────────────────────────────────────────────
    FactRule(
        "RETURNED",
        re.compile(
            r"odesłał(?:em|am)?\s+towar|towar\s+(?:został\s+)?odesłany|"
            r"zwróciłem?\s+towar",
            re.I,
        ),
        [("ORDER", "ORDER"), ("RETURN_SHIPMENT", "AUTO_RETURN_ORDER"), ("DATE", "DATE")],
    ),
    FactRule(
        "RETURN_PROOF_PROVIDED",
        re.compile(
            r"(?:mam|dostarczyłem?|wysłałem?)\s+(?:dowód|potwierdzenie)\s+"
            r"(?:nadania|odesłania)",
            re.I,
        ),
        [("ORDER", "ORDER"), ("DATE", "DATE")],
    ),
    # ── Reklamacja ───────────────────────────────────────────────────────────
    FactRule(
        "COMPLAINT_SUBMITTED",
        re.compile(
            r"zgłosił(?:em|am)?\s+reklamacj[ęę]|złożył(?:em|am)?\s+reklamacj[ęę]",
            re.I,
        ),
        [
            ("CUSTOMER", "IMPLICIT_CUSTOMER"), ("ORDER", "ORDER"),
            ("COMPLAINT", "COMPLAINT"), ("DATE", "DATE"),
        ],
    ),
    FactRule(
        "COMPLAINT_RESPONSE_SENT",
        re.compile(
            r"sklep\s+(?:odpowiedział|wysłał\s+odpowiedź|udzielił\s+odpowiedzi)",
            re.I,
        ),
        [("STORE", "IMPLICIT_STORE"), ("COMPLAINT", "COMPLAINT"), ("DATE", "DATE")],
    ),
    # ── Konto ────────────────────────────────────────────────────────────────
    FactRule(
        "ACCOUNT_BLOCKED",
        re.compile(
            r"sklep\s+zablokował\s+(?:moje\s+)?konto|"
            r"konto\s+zostało\s+zablokowane\s+(?:przez\s+sklep)?",
            re.I,
        ),
        [
            ("STORE", "IMPLICIT_STORE"), ("ACCOUNT", "ACCOUNT"),
            ("DATE", "DATE"), ("REASON", "literal:FRAUD_SUSPECT"),
        ],
    ),
    # ── Chargeback ───────────────────────────────────────────────────────────
    FactRule(
        "CHARGEBACK_OPENED",
        re.compile(
            r"otworzyłem\s+chargeback|złożyłem\s+chargeback|"
            r"zgłosiłem\s+chargeback",
            re.I,
        ),
        [
            ("CUSTOMER", "IMPLICIT_CUSTOMER"), ("ORDER", "ORDER"),
            ("CHARGEBACK", "CHARGEBACK"), ("DATE", "DATE"),
        ],
    ),
    FactRule(
        "CHARGEBACK_RESOLVED",
        re.compile(
            r"bank\s+uznał\s+(?:chargeback\s+)?na\s+(?:moją\s+)?korzyść|"
            r"wygrał\s+(?:chargeback\s+)?klient",
            re.I,
        ),
        [("CHARGEBACK", "CHARGEBACK"), ("DATE", "DATE"), ("RESULT", "literal:WON_BY_CUSTOMER")],
    ),
    FactRule(
        "CHARGEBACK_RESOLVED",
        re.compile(
            r"bank\s+uznał\s+(?:reklamację\s+)?sklepu|wygrał\s+sklep|"
            r"przegrał\s+klient|chargeback\s+przegrany\s+przez\s+klienta",
            re.I,
        ),
        [("CHARGEBACK", "CHARGEBACK"), ("DATE", "DATE"), ("RESULT", "literal:WON_BY_STORE")],
    ),
    # ── Kupony ───────────────────────────────────────────────────────────────
    FactRule(
        "COUPON_APPLIED",
        re.compile(
            r"użył(?:em|am)?\s+(?:dwóch\s+)?kupon(?:ów|y|u)|"
            r"zastosował(?:em|am)?\s+(?:kod|kupon)|"
            r"próbował(?:em|am)?\s+użyć\s+(?:dwóch\s+)?kupon",
            re.I,
        ),
        [
            ("CUSTOMER", "IMPLICIT_CUSTOMER"), ("ORDER", "ORDER"),
            ("COUPON", "COUPON"), ("DATE", "DATE"),
        ],
    ),
]
