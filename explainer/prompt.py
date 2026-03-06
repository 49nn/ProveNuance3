"""
prompt.py — Budowanie promptów dla LLM Explainer.

Eksportuje:
    build_system_prompt(language) → str
    build_user_message(case_text, facts, proof_run, cluster_states, entity_map,
                       neural_trace, grounded) → str

Proweniencja w user_message:
  - Symboliczna: proof_dag (depends_on + naf_checked) + used_fact_ids z ProofStep
  - Tekstowa:    source_id przy każdym fakcie (skąd pochodzi obserwacja)
  - Neuronalna:  neural_trace — dict[fact_id, list[NeuralTraceItem]] (top-k wpływów NN)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from data_model.fact import Fact, FactStatus, NeuralTraceItem

if TYPE_CHECKING:
    from data_model.cluster import ClusterStateRow
    from sv.proof import ProofRun

# ---------------------------------------------------------------------------
# Mapowania czytelnych nazw predykatów i klastrów (PL)
# ---------------------------------------------------------------------------

_PREDICATE_PL: dict[str, str] = {
    # Fakty obserwowane z ekstrakcji LLM
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
właściwości stron, stwierdzone fakty, (opcjonalnie) dowód logiczny i ślad sieci neuronowej.

Twoim zadaniem jest:
1. Wyjaśnić sytuację prawną w sposób jasny i zrozumiały dla człowieka.
2. Wskazać kluczowe fakty mające wpływ na ocenę sprawy.
3. Podać wniosek: jakie prawa i obowiązki wynikają z ustalonych faktów.
4. Wyjaśnić PROWENIENCJĘ wniosków — skąd system wywnioskował dane konkluzje:
   - jeśli dostępny jest Dowód logiczny: wskaż które fakty były przesłankami (sekcja "bo:"),
     które warunki musiały NIE zachodzić (sekcja "brak:"), i jaką regułą posłużył się system;
   - jeśli dostępny jest Ślad sieci neuronowej: opisz które właściwości stron lub inne fakty
     wzmocniły (+) lub osłabiły (−) ocenę poszczególnych faktów;
   - jeśli wynik dowodu to "unknown" lub "not_proved": wyjaśnij wprost, że system nie znalazł
     pasujących reguł i ocena opiera się wyłącznie na zaobserwowanych faktach.

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
party attributes, established facts, an optional logical proof, and an optional neural trace.

Your task:
1. Explain the legal situation in a clear, human-readable way.
2. Highlight the key facts affecting the legal assessment.
3. State the conclusion: what rights and obligations follow from the established facts.
4. Explain the PROVENANCE of conclusions — how the system reached each conclusion:
   - if a Logical proof is available: cite which facts were premises (the "because:" lines),
     which conditions had to be absent (the "absent:" lines), and which rule was applied;
   - if a Neural trace is available: describe which party attributes or facts strengthened (+)
     or weakened (−) the assessment of individual facts;
   - if the proof result is "unknown" or "not_proved": state explicitly that no matching rules
     were found and the assessment is based solely on observed facts.

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
    cluster_domains: dict[str, list[str]] | None = None,
    grounded: bool = False,
    neural_trace: dict[str, list[NeuralTraceItem]] | None = None,
) -> str:
    """
    Buduje wiadomość użytkownika dla Gemini.

    Sekcje:
      1. Opis sprawy (oryginalny tekst) — pomijane gdy grounded=True
      2. Właściwości stron (cluster_states)
      3. Stwierdzone fakty (proved > observed) + tag source_id
      4. Dowód logiczny (proof_dag z depends_on + naf_checked + used_fact_ids)
      5. Ślad neuronalny (neural_trace — top-k wpływów NN per fakt, opcjonalnie)
      6. Pytanie końcowe

    grounded=True: nie wysyła tekstu sprawy — LLM opiera się wyłącznie
    na danych strukturalnych (faktach i klastrach).
    """
    sections: list[str] = []

    # 1. Opis sprawy (tylko w trybie niegrounded)
    if not grounded:
        sections.append(f"## Opis sprawy\n\n{case_text.strip()}")

    # 2. Właściwości stron
    if cluster_states:
        lines = [_render_cluster(cs, entity_map, cluster_domains) for cs in cluster_states]
        sections.append("## Właściwości stron\n\n" + "\n".join(lines))

    # 3. Stwierdzone fakty (z source_id jako proweniencja tekstowa)
    proved = [f for f in facts if f.status == FactStatus.proved]
    observed = [f for f in facts if f.status == FactStatus.observed]
    all_visible = proved + observed  # proved najpierw

    if all_visible:
        fact_lines = [_render_fact(f, entity_map) for f in all_visible]
        header = "## Stwierdzone fakty"
        if proved:
            header += f" ({len(proved)} udowodnionych, {len(observed)} zaobserwowanych)"
        sections.append(header + "\n\n" + "\n".join(fact_lines))

    # 4. Dowód logiczny (symboliczna proweniencja)
    if proof_run is not None:
        proof_block = _render_proof_steps(proof_run, facts)
        sections.append("## Dowód logiczny\n\n" + proof_block)

    # 5. Ślad neuronalny (proweniencja NN)
    if neural_trace:
        nn_block = _render_neural_trace(neural_trace, facts, entity_map)
        sections.append("## Ślad sieci neuronowej\n\n" + nn_block)

    # 6. Pytanie
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
    """
    Konwertuje Fact na czytelną linię tekstową.
    Proweniencja tekstowa: source_id z fact.source dołączany jako tag "← source_id".
    """
    label = _PREDICATE_PL.get(fact.predicate, fact.predicate)
    status_marker = "✓" if fact.status == FactStatus.proved else "·"

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

    # Proweniencja tekstowa — skąd pochodzi obserwacja
    source_tag = ""
    if fact.source and fact.source.source_id:
        source_tag = f"  ← {fact.source.source_id}"

    body = f"{label}: {args_str}" if args_str else label
    return f"{status_marker} {body}{source_tag}"


def _render_cluster(
    cs: "ClusterStateRow",
    entity_map: dict[str, str],
    cluster_domains: dict[str, list[str]] | None = None,
) -> str:
    """Konwertuje ClusterStateRow na czytelną linię."""
    name = entity_map.get(cs.entity_id, cs.entity_id)
    cluster_label = _CLUSTER_PL.get(cs.cluster_name, cs.cluster_name)

    # Wyznacz wartość z najwyższym logitem
    if cs.logits:
        domain = (cluster_domains or {}).get(cs.cluster_name)
        if domain and len(cs.logits) == len(domain):
            best_idx = max(range(len(cs.logits)), key=lambda i: cs.logits[i])
            raw_value = domain[best_idx]
            value = _VALUE_PL.get(raw_value, raw_value)
        else:
            value = "?"
    else:
        value = "?"

    return f"  - {name}: {cluster_label} = {value}"


def _render_proof_steps(proof_run: "ProofRun", facts: list[Fact]) -> str:
    """
    Renderuje udowodnione konkluzje z proof_dag do czytelnej formy tekstowej.

    Proweniencja symboliczna:
      - depends_on: atomy-przesłanki pozytywne (FIX: było "body_atoms")
      - naf_checked: warunki które nie miały zachodzić (NAF)
      - used_fact_ids: mapowanie na czytelne etykiety faktów wejściowych
    """
    if proof_run.result in ("not_proved", "unknown"):
        n_base = sum(1 for e in proof_run.proof_dag if isinstance(e, dict) and e.get("rule_id") is None)
        msg = "Żadna reguła nie wyprowadzi konkluzji dla tej sprawy."
        if n_base:
            msg += f" System ocenił {n_base} fakt(ów) bazowych, lecz nie znalazł pasujących przesłanek reguł."
        return msg

    # Indeks fact_id -> Fact (do mapowania used_fact_ids w ProofStep)
    fact_index: dict[str, Fact] = {f.fact_id: f for f in facts}

    # Preferowane: proof_dag (pełna proweniencja per atom)
    dag_entries = [e for e in proof_run.proof_dag if isinstance(e, dict) and "atom" in e]
    # Pomiń atomy bazowe (rule_id=None) — to fakty wejściowe, nie konkluzje
    derived = [e for e in dag_entries if e.get("rule_id") is not None]
    if derived:
        lines: list[str] = [f"Wynik dowodu: **{proof_run.result}**\n"]
        for entry in derived:
            atom_str: str = entry["atom"]  # np. "may_cancel_for_nonpayment(o300)"
            predicate = atom_str.split("(")[0].upper()
            atom_label = _PREDICATE_PL.get(predicate, predicate.replace("_", " ").title())

            # Przesłanki pozytywne — "depends_on" (nowy format) lub "body_atoms" (stary DB)
            depends_on: list[str] = entry.get("depends_on") or entry.get("body_atoms") or []
            body_labels = [_atom_to_label(a) for a in depends_on if a]

            # Warunki NAF — "naf_checked" (nowy) lub "naf_atoms" (stary DB)
            naf_checked: list[str] = entry.get("naf_checked") or entry.get("naf_atoms") or []
            naf_labels = [_atom_to_label(a) for a in naf_checked if a]

            line = f"  - Udowodniono: **{atom_label}**"
            if body_labels:
                line += f"\n    bo: {'; '.join(body_labels)}"
            if naf_labels:
                line += f"\n    brak: {'; '.join(naf_labels)}"
            lines.append(line)
        return "\n".join(lines)

    # Fallback: proof_dag pusty — użyj ProofStep z used_fact_ids
    if not proof_run.steps:
        return f"Wynik: {proof_run.result}"

    lines = [f"Wynik dowodu: **{proof_run.result}**\n"]
    for step in proof_run.steps:
        if step.rule_id is None:
            continue
        rule_name = step.rule_id.split(".")[-1].upper()
        rule_label = _PREDICATE_PL.get(rule_name, step.rule_id)

        # Mapuj used_fact_ids na czytelne etykiety faktów
        premise_labels: list[str] = []
        for fid in step.used_fact_ids:
            f = fact_index.get(fid)
            if f:
                premise_labels.append(_PREDICATE_PL.get(f.predicate, f.predicate))

        subst_parts = [f"{k}={v}" for k, v in step.substitution.items()]
        subst_str = ", ".join(subst_parts)

        line = f"  - Reguła: {rule_label}"
        if subst_str:
            line += f" (podstawienie: {subst_str})"
        if premise_labels:
            line += f"\n    przesłanki: {'; '.join(premise_labels)}"
        lines.append(line)
    return "\n".join(lines)


def _render_neural_trace(
    neural_trace: dict[str, list[NeuralTraceItem]],
    facts: list[Fact],
    entity_map: dict[str, str],
) -> str:
    """
    Renderuje ślad neuronalny — top-k wpływów message-passing per fakt docelowy.

    Format per fakt:
      [Nazwa faktu]
        + typ_krawędzi: klaster (encja), krok N  <- wpływ zwiększający P(T)
        - typ_krawędzi: klaster (encja), krok N  <- wpływ zmniejszający P(T)

    neural_trace: dict[target_fact_id, list[NeuralTraceItem]] (posortowane wg magnitude)
    """
    fact_index: dict[str, Fact] = {f.fact_id: f for f in facts}
    lines: list[str] = []

    for fact_id, trace_items in neural_trace.items():
        if not trace_items:
            continue
        fact = fact_index.get(fact_id)
        fact_label = _PREDICATE_PL.get(fact.predicate, fact.predicate) if fact else fact_id

        item_lines: list[str] = []
        for item in trace_items[:3]:  # top-3 na fakt (zeby nie zasmiecac promptu)
            delta_T = item.delta_logits.get("T", 0.0)
            direction = "+" if delta_T >= 0 else "-"

            if item.from_cluster_id:
                # format: "{cluster_name}:{entity_id}"
                cluster_name, _, entity_id = item.from_cluster_id.partition(":")
                cluster_label = _CLUSTER_PL.get(cluster_name, cluster_name)
                entity_name = entity_map.get(entity_id, entity_id)
                item_lines.append(
                    f"    {direction} {item.edge_type}: {cluster_label} ({entity_name}), krok {item.step}"
                )
            elif item.from_fact_id:
                src_fact = fact_index.get(item.from_fact_id)
                src_label = (
                    _PREDICATE_PL.get(src_fact.predicate, src_fact.predicate)
                    if src_fact else item.from_fact_id
                )
                item_lines.append(
                    f"    {direction} {item.edge_type}: <- {src_label}, krok {item.step}"
                )

        if item_lines:
            lines.append(f"  [{fact_label}]")
            lines.extend(item_lines)

    if not lines:
        return "Brak danych sledowych z sieci neuronowej."
    return "\n".join(lines)
