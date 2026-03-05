"""
prompt.py — Budowanie promptów dla LLM Explainer.

Eksportuje:
    build_system_prompt(language) → str
    build_user_message(case_text, facts, proof_run, cluster_states, entity_map) → str
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from data_model.fact import Fact, FactStatus

if TYPE_CHECKING:
    from nn.graph_builder import ClusterStateRow
    from sv.proof import ProofRun

# ---------------------------------------------------------------------------
# Mapowania czytelnych nazw predykatów i klastrów (PL)
# ---------------------------------------------------------------------------

_PREDICATE_PL: dict[str, str] = {
    # Fakty obserwowane (TextExtractor / LLMExtractor)
    "ORDER_PLACED":                    "Zamówienie złożone",
    "ORDER_ACCEPTED":                  "Zamówienie przyjęte",
    "ORDER_CANCELLED":                 "Zamówienie anulowane",
    "PAYMENT_MADE":                    "Płatność dokonana",
    "DELIVERED":                       "Towar dostarczony",
    "WITHDRAWAL_STATEMENT_SUBMITTED":  "Oświadczenie o odstąpieniu złożone",
    "RETURNED":                        "Towar zwrócony",
    "REFUND_ISSUED":                   "Zwrot środków",
    "COMPLAINT_FILED":                 "Reklamacja złożona",
    "COMPLAINT_ACCEPTED":              "Reklamacja uznana",
    "COMPLAINT_REJECTED":              "Reklamacja odrzucona",
    "ACCOUNT_BLOCKED":                 "Konto zablokowane",
    "CHARGEBACK_OPENED":               "Chargeback otwarty",
    "CHARGEBACK_RESOLVED":             "Chargeback rozstrzygnięty",
    "COUPON_APPLIED":                  "Kupon zastosowany",
    "CONTRACT_FORMED":                 "Umowa zawarta",
    "CONSUMER_WITHDRAWAL_RIGHT":       "Prawo do odstąpienia (konsument)",
    "COMPLAINT_ACCEPTED_RULE":         "Reguła: reklamacja uznana",
    "DOWNLOAD_STARTED":                "Pobieranie rozpoczęte",
    # Konkluzje SV (derived predicates z reguł)
    "MAY_CANCEL_FOR_NONPAYMENT":       "Sklep może anulować za brak płatności",
    "PREPAID":                         "Zamówienie opłacone z góry",
    "AB_CUSTOMER_PAYS_RETURN":         "Klient pokrywa koszty zwrotu",
    "AB_STORE_PAYS_RETURN":            "Sklep pokrywa koszty zwrotu",
    "WITHDRAWAL_ALLOWED":              "Odstąpienie od umowy dopuszczalne",
    "WITHDRAWAL_BLOCKED":              "Odstąpienie od umowy niedopuszczalne",
    "REFUND_DUE":                      "Należy się zwrot środków",
    "COMPLAINT_VALID":                 "Reklamacja zasadna",
    "COMPLAINT_INVALID":               "Reklamacja bezzasadna",
    "ACCOUNT_SUSPENSION_VALID":        "Blokada konta zasadna",
    "CHARGEBACK_VALID":                "Chargeback zasadny",
}

_CLUSTER_PL: dict[str, str] = {
    "customer_type":          "typ klienta",
    "order_status":           "status zamówienia",
    "payment_method":         "metoda płatności",
    "product_type":           "typ produktu",
    "defective":              "produkt wadliwy",
    "store_pays_return":      "sklep pokrywa zwrot",
    "digital_consent":        "zgoda na treść cyfrową",
    "download_started_flag":  "pobieranie rozpoczęte",
    "password_shared":        "hasło udostępnione",
    "coupon_stackable":       "kupon łączowalny",
    "account_status":         "status konta",
    "chargeback_status":      "status chargeback",
}

_VALUE_PL: dict[str, str] = {
    "CONSUMER":            "konsument",
    "BUSINESS":            "przedsiębiorca",
    "PLACED":              "złożone",
    "ACCEPTED":            "przyjęte",
    "PAID":                "opłacone",
    "CANCELLED":           "anulowane",
    "DELIVERED":           "dostarczone",
    "DISPUTED":            "sporne",
    "CARD":                "karta",
    "TRANSFER":            "przelew",
    "BLIK":                "BLIK",
    "COD":                 "za pobraniem",
    "PHYSICAL":            "fizyczny",
    "DIGITAL":             "cyfrowy",
    "CUSTOM":              "na zamówienie",
    "YES":                 "tak",
    "NO":                  "nie",
    "UNKNOWN":             "nieznane",
    "ACTIVE":              "aktywne",
    "BLOCKED":             "zablokowane",
    "NONE":                "brak",
    "OPEN":                "otwarty",
    "RESOLVED_CUSTOMER":   "rozstrzygnięty na korzyść klienta",
    "RESOLVED_STORE":      "rozstrzygnięty na korzyść sklepu",
}

_SYSTEM_PL = """\
Jesteś asystentem analizującym spory e-commerce na podstawie dostarczonych danych.

Otrzymasz opis sprawy oraz dane wyekstrahowane przez system automatycznej analizy:
właściwości stron, stwierdzone fakty i (opcjonalnie) dowód logiczny.

Twoim zadaniem jest:
1. Wyjaśnić sytuację prawną w sposób jasny i zrozumiały dla człowieka.
2. Wskazać kluczowe fakty mające wpływ na ocenę sprawy.
3. Podać wniosek: jakie prawa i obowiązki wynikają z ustalonych faktów.

WAŻNE OGRANICZENIA:
- Opieraj się WYŁĄCZNIE na faktach i właściwościach zawartych w sekcjach poniżej.
- Nie stosuj własnej wiedzy prawnej ani domysłów wykraczających poza dostarczone dane.
- Jeśli jakiś fakt nie jest zawarty w danych — nie zakładaj go ani nie wnioskuj o nim.
- Jeśli dane są niewystarczające do oceny — napisz to wprost.

Odpowiadaj po polsku. Unikaj żargonu technicznego (np. nazw predykatów, identyfikatorów).
Używaj naturalnych określeń: "klient", "sklep", "zamówienie", dat zamiast kodów encji.
Bądź rzeczowy i konkretny — odpowiedź powinna mieć 3–6 akapitów.
"""

_SYSTEM_EN = """\
You are an assistant analyzing e-commerce disputes based on the provided data.

You will receive a case description and data extracted by an automated analysis system:
party attributes, established facts, and (optionally) a logical proof.

Your task:
1. Explain the legal situation in a clear, human-readable way.
2. Highlight the key facts affecting the legal assessment.
3. State the conclusion: what rights and obligations follow from the established facts.

IMPORTANT CONSTRAINTS:
- Base your analysis SOLELY on the facts and attributes provided in the sections below.
- Do not apply your own legal knowledge or make inferences beyond the provided data.
- If a fact is not present in the data — do not assume or infer it.
- If the data is insufficient for a conclusion — state that explicitly.

Answer in English. Avoid technical jargon (predicate names, entity IDs).
Use natural terms: "customer", "store", "order", dates instead of entity codes.
Be concise and specific — 3–6 paragraphs.
"""


# ---------------------------------------------------------------------------
# Publiczne API
# ---------------------------------------------------------------------------

def build_system_prompt(language: str = "pl") -> str:
    """Zwraca system prompt w żądanym języku."""
    return _SYSTEM_PL if language == "pl" else _SYSTEM_EN


def build_user_message(
    case_text: str,
    facts: list[Fact],
    proof_run: "ProofRun | None",
    cluster_states: "list[ClusterStateRow] | None",
    entity_map: dict[str, str],
    grounded: bool = False,
) -> str:
    """
    Buduje wiadomość użytkownika dla Gemini.

    Sekcje:
      1. Opis sprawy (oryginalny tekst) — pomijane gdy grounded=True
      2. Właściwości stron (cluster_states)
      3. Stwierdzone fakty (proved > observed)
      4. Dowód logiczny (ProofRun.steps, opcjonalnie)
      5. Pytanie końcowe

    grounded=True: nie wysyła tekstu sprawy — LLM opiera się wyłącznie
    na danych strukturalnych (faktach i klastrach).
    """
    sections: list[str] = []

    # 1. Opis sprawy (tylko w trybie niegrounded)
    if not grounded:
        sections.append(f"## Opis sprawy\n\n{case_text.strip()}")

    # 2. Właściwości stron
    if cluster_states:
        lines = [_render_cluster(cs, entity_map) for cs in cluster_states]
        sections.append("## Właściwości stron\n\n" + "\n".join(lines))

    # 3. Stwierdzone fakty
    visible_statuses = {FactStatus.proved, FactStatus.observed}
    proved = [f for f in facts if f.status == FactStatus.proved]
    observed = [f for f in facts if f.status == FactStatus.observed]
    all_visible = proved + observed  # proved najpierw

    if all_visible:
        fact_lines = [_render_fact(f, entity_map) for f in all_visible]
        header = "## Stwierdzone fakty"
        if proved:
            header += f" ({len(proved)} udowodnionych, {len(observed)} zaobserwowanych)"
        sections.append(header + "\n\n" + "\n".join(fact_lines))

    # 4. Dowód logiczny
    if proof_run is not None:
        proof_block = _render_proof_steps(proof_run)
        sections.append("## Dowód logiczny\n\n" + proof_block)

    # 5. Pytanie
    sections.append(
        "## Pytanie\n\n"
        "Na podstawie powyższych danych wyjaśnij sytuację prawną klienta i sklepu. "
        "Wskaż kluczowe fakty, obowiązujące przepisy oraz końcowy wniosek."
    )

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Renderowanie pojedynczych elementów
# ---------------------------------------------------------------------------

def _atom_to_label(atom_str: str) -> str:
    """
    Zamienia atom Clingo (np. 'payment_method(transfer,o300)') na czytelny tekst.
    Przykład: 'order_placed(cust1,d_2026_03_03,o300)' → 'Zamówienie złożone (cust1, 2026-03-03, o300)'
    """
    if "(" in atom_str:
        pred_raw, rest = atom_str.split("(", 1)
        args_raw = rest.rstrip(")")
        args = [a.strip() for a in args_raw.split(",") if a.strip()]
    else:
        pred_raw, args = atom_str, []

    pred_upper = pred_raw.upper()
    label = _PREDICATE_PL.get(pred_upper, pred_raw.replace("_", " ").title())

    # Czytelne argumenty: d_2026_03_03 → 2026-03-03, reszta bez zmian
    readable_args = []
    for a in args:
        if a.startswith("d_"):
            readable_args.append(a[2:].replace("_", "-"))
        else:
            readable_args.append(a)

    if readable_args:
        return f"{label} ({', '.join(readable_args)})"
    return label


def _render_fact(fact: Fact, entity_map: dict[str, str]) -> str:
    """Konwertuje Fact na czytelną linię tekstową."""
    label = _PREDICATE_PL.get(fact.predicate, fact.predicate)
    status_marker = "✓" if fact.status == FactStatus.proved else "·"

    args_parts: list[str] = []
    for arg in fact.args:
        if arg.entity_id:
            name = entity_map.get(arg.entity_id, arg.entity_id)
            args_parts.append(name)
        elif arg.literal_value:
            args_parts.append(arg.literal_value)

    # Data (encje D_*) — wyciągnij czytelnie
    readable_args: list[str] = []
    date_parts: list[str] = []
    for arg in fact.args:
        if arg.entity_id and arg.entity_id.startswith("D_"):
            date_parts.append(arg.entity_id[2:])  # "D_2026-03-01" → "2026-03-01"
        elif arg.entity_id:
            name = entity_map.get(arg.entity_id, arg.entity_id)
            readable_args.append(name)
        elif arg.literal_value:
            readable_args.append(arg.literal_value)

    parts = readable_args
    if date_parts:
        parts = parts + [f"[{', '.join(date_parts)}]"]

    args_str = ", ".join(parts) if parts else ""
    return f"{status_marker} {label}: {args_str}" if args_str else f"{status_marker} {label}"


def _render_cluster(cs: "ClusterStateRow", entity_map: dict[str, str]) -> str:
    """Konwertuje ClusterStateRow na czytelną linię."""
    name = entity_map.get(cs.entity_id, cs.entity_id)
    cluster_label = _CLUSTER_PL.get(cs.cluster_name, cs.cluster_name)

    # Wyznacz wartość z najwyższym logitem
    schema_domain: list[str] | None = None
    # Jeśli klaster ma logity — wybierz argmax
    if cs.logits:
        # Domyślne domeny z ontologii (kolejność z seed_ontology.sql)
        _DOMAIN_MAP: dict[str, list[str]] = {
            "customer_type":         ["CONSUMER", "BUSINESS"],
            "order_status":          ["PLACED", "ACCEPTED", "PAID", "CANCELLED", "DELIVERED", "DISPUTED"],
            "payment_method":        ["CARD", "TRANSFER", "BLIK", "COD"],
            "product_type":          ["PHYSICAL", "DIGITAL", "CUSTOM"],
            "defective":             ["YES", "NO", "UNKNOWN"],
            "store_pays_return":     ["YES", "NO", "UNKNOWN"],
            "digital_consent":       ["YES", "NO", "UNKNOWN"],
            "download_started_flag": ["YES", "NO", "UNKNOWN"],
            "password_shared":       ["YES", "NO", "UNKNOWN"],
            "coupon_stackable":      ["YES", "NO"],
            "account_status":        ["ACTIVE", "BLOCKED"],
            "chargeback_status":     ["NONE", "OPEN", "RESOLVED_CUSTOMER", "RESOLVED_STORE"],
        }
        domain = _DOMAIN_MAP.get(cs.cluster_name)
        if domain and len(cs.logits) == len(domain):
            best_idx = max(range(len(cs.logits)), key=lambda i: cs.logits[i])
            raw_value = domain[best_idx]
            value = _VALUE_PL.get(raw_value, raw_value)
        else:
            value = "?"
    else:
        value = "?"

    return f"  - {name}: {cluster_label} = {value}"


def _render_proof_steps(proof_run: "ProofRun") -> str:
    """Renderuje udowodnione konkluzje z proof_dag do czytelnej formy tekstowej."""
    if proof_run.result == "not_proved":
        return "Nie udowodniono żadnej konkluzji logicznej."

    # Renderuj z proof_dag (preferowane — zawiera atom + rule_id + body_atoms)
    dag_entries = [e for e in proof_run.proof_dag if isinstance(e, dict) and "atom" in e]
    # Pomiń atomy bazowe (status=base, rule_id=None) — to fakty wejściowe, nie konkluzje
    derived = [e for e in dag_entries if e.get("rule_id") is not None]
    if derived:
        lines: list[str] = [f"Wynik dowodu: **{proof_run.result}**\n"]
        for entry in derived:
            atom_str: str = entry["atom"]  # np. "may_cancel_for_nonpayment(o300)"
            predicate = atom_str.split("(")[0].upper()
            atom_label = _PREDICATE_PL.get(predicate, predicate.replace("_", " ").title())

            # Proweniencja: jakie atomy były przesłankami
            body: list[str] = entry.get("body_atoms") or []
            body_labels = [_atom_to_label(a) for a in body if a]

            line = f"  - Udowodniono: **{atom_label}**"
            if body_labels:
                line += f"\n    bo: {'; '.join(body_labels)}"
            lines.append(line)
        return "\n".join(lines)

    # Fallback: jeśli dag pusty, użyj steps
    if not proof_run.steps:
        return f"Wynik: {proof_run.result}"

    lines = [f"Wynik dowodu: **{proof_run.result}**\n"]
    for step in proof_run.steps:
        if step.rule_id is None:
            continue
        rule_name = step.rule_id.split(".")[-1].upper()
        rule_label = _PREDICATE_PL.get(rule_name, step.rule_id)
        subst_parts = [f"{k}={v}" for k, v in step.substitution.items()]
        subst_str = ", ".join(subst_parts)
        lines.append(f"  - Reguła: {rule_label}" +
                     (f" (podstawienie: {subst_str})" if subst_str else ""))
    return "\n".join(lines)
