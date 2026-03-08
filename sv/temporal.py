"""
Constraints temporalne dla Symbolic Verifier.

Dwa rodzaje:

TemporalConstraint — wymaganie kolejności:
  earlier_pred.earlier_date < later_pred.later_date
  Naruszenie gdy later_date < earlier_date (ścisłe before).
  Ten sam czas NIE jest naruszeniem.

  Generuje:
    temporal_violation(name, Key) :-
        order_placed(..., Key, EarlyDate),
        return_request(..., Key, LateDate),
        before(LateDate, EarlyDate).

TemporalCoincidenceConstraint — wymaganie współwystępowania w tym samym okresie:
  oba zdarzenia muszą mieć tę samą datę na poziomie: day / week / month / year.
  Naruszenie gdy NOT same_<period>(DA, DB).

  Generuje (NAF, stratum=1):
    temporal_violation(name, Key) :-
        pred_a(..., Key, DA),
        pred_b(..., Key, DB),
        not same_month(DA, DB).

Uwaga: same_* są safe dla NAF bo są wyłącznie computed base facts —
stratification validator pomija predykaty bez reguł definiujących.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from data_model.common import ConstTerm, RuleArg, VarTerm
from data_model.rule import LiteralType, Rule, RuleBodyLiteral, RuleHead, RuleLanguage, RuleMetadata

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalConstraint:
    """
    Constraint temporalny: earlier_pred musi wystąpić przed later_pred.

    Pola:
      name             — unikalny identyfikator, trafia jako arg do violation_atom
      earlier_pred     — predykat zdarzenia wcześniejszego (np. "order_placed")
      earlier_key_role — nazwa roli klucza w earlier_pred (np. "ORDER")
      earlier_date_role— nazwa roli daty w earlier_pred (np. "DATE")
      later_pred       — predykat zdarzenia późniejszego (np. "return_request")
      later_key_role   — nazwa roli klucza w later_pred (np. "ORDER")
      later_date_role  — nazwa roli daty w later_pred (np. "DATE")
      violation_atom   — predykat naruszenia (domyślnie "temporal_violation")
    """

    name: str
    earlier_pred: str
    earlier_key_role: str
    earlier_date_role: str
    later_pred: str
    later_key_role: str
    later_date_role: str
    violation_atom: str = field(default="temporal_violation")


# Granulacje czasu obsługiwane przez same_<period> computed facts.
TemporalPeriod = Literal["day", "week", "month", "year"]

_PERIOD_PREDICATE: dict[str, str] = {
    "day":   "same_day",
    "week":  "same_week",
    "month": "same_month",
    "year":  "same_year",
}

# Pozycje ról dla predykatów same_* i before (dodawane do all_positions w verifier).
TEMPORAL_HELPER_POSITIONS: dict[str, list[str]] = {
    "before":     ["FROM", "TO"],
    "same_day":   ["FROM", "TO"],
    "same_week":  ["FROM", "TO"],
    "same_month": ["FROM", "TO"],
    "same_year":  ["FROM", "TO"],
}

# Predykaty pomocnicze temporalne — nie eksponujemy jako Fact.
TEMPORAL_HELPER_PREDICATES: frozenset[str] = frozenset(TEMPORAL_HELPER_POSITIONS)


@dataclass(frozen=True)
class TemporalCoincidenceConstraint:
    """
    Constraint temporalny: oba zdarzenia muszą wystąpić w tym samym okresie.

    Pola:
      name         — unikalny identyfikator (np. "delivery_same_month_as_payment")
      pred_a       — pierwszy predykat (np. "payment_made")
      key_role_a   — rola klucza w pred_a (np. "ORDER")
      date_role_a  — rola daty w pred_a (np. "DATE")
      pred_b       — drugi predykat (np. "delivery")
      key_role_b   — rola klucza w pred_b (np. "ORDER")
      date_role_b  — rola daty w pred_b (np. "DATE")
      period       — granulacja: "day" | "week" | "month" | "year"
      violation_atom — predykat naruszenia (domyślnie "temporal_violation")

    Generuje regułę Horn+NAF (stratum=1):
      violation_atom(name, Key) :-
          pred_a(..., Key, DA),
          pred_b(..., Key, DB),
          not same_<period>(DA, DB).
    """

    name: str
    pred_a: str
    key_role_a: str
    date_role_a: str
    pred_b: str
    key_role_b: str
    date_role_b: str
    period: TemporalPeriod
    violation_atom: str = field(default="temporal_violation")


@dataclass(frozen=True)
class TemporalWindowConstraint:
    """
    Constraint temporalny: późniejsze zdarzenie musi nastąpić w ciągu N dni od wcześniejszego.

    Pola:
      name             — unikalny identyfikator (np. "return_within_14_days")
      earlier_pred     — predykat zdarzenia referencyjnego (np. "order_placed")
      earlier_key_role — rola klucza w earlier_pred (np. "ORDER")
      earlier_date_role— rola daty w earlier_pred (np. "DATE")
      later_pred       — predykat zdarzenia które musi być w oknie (np. "return_request")
      later_key_role   — rola klucza w later_pred (np. "ORDER")
      later_date_role  — rola daty w later_pred (np. "DATE")
      n_days           — maksymalna liczba dni po earlier_date (włącznie; >= 0)
      violation_atom   — predykat naruszenia (domyślnie "temporal_violation")

    Naruszenie gdy later_date NIE jest w oknie [earlier_date, earlier_date + n_days].
    Jeśli later_date < earlier_date — też naruszenie (niejawnie przez brak within_N_days_after).

    Generuje (NAF, stratum=1):
      violation_atom(name, Key) :-
          earlier_pred(..., Key, DA),
          later_pred(..., Key, DB),
          not within_{n_days}_days_after(DA, DB).

    Computed facts within_{n}_days_after(DA, DB) są generowane przez _extract_computed_facts
    dla wszystkich par dat gdzie 0 <= (DB - DA).days <= n.
    """

    name: str
    earlier_pred: str
    earlier_key_role: str
    earlier_date_role: str
    later_pred: str
    later_key_role: str
    later_date_role: str
    n_days: int
    violation_atom: str = field(default="temporal_violation")

    def window_predicate(self) -> str:
        """Nazwa Clingo predykatu dla okna this constraint."""
        return f"within_{self.n_days}_days_after"


AnyTemporalConstraint = TemporalConstraint | TemporalCoincidenceConstraint | TemporalWindowConstraint


def window_predicate_name(n_days: int) -> str:
    """Zwraca nazwę predykatu okna dla danego n_days."""
    return f"within_{n_days}_days_after"


def temporal_constraints_to_rules(
    constraints: list[AnyTemporalConstraint],
    predicate_positions: dict[str, list[str]],
) -> list[Rule]:
    """
    Konwertuje listę TemporalConstraint / TemporalCoincidenceConstraint do Rule obiektów.

    Predykaty nieobecne w predicate_positions są pomijane z ostrzeżeniem.
    Wygenerowane reguły trafiają do programu Clingo tak samo jak reguły domenowe.
    """
    rules: list[Rule] = []
    for tc in constraints:
        if isinstance(tc, TemporalCoincidenceConstraint):
            rule = _tcc_to_rule(tc, predicate_positions)
        elif isinstance(tc, TemporalWindowConstraint):
            rule = _tw_to_rule(tc, predicate_positions)
        else:
            rule = _tc_to_rule(tc, predicate_positions)
        if rule is not None:
            rules.append(rule)
        else:
            log.warning(
                "TemporalConstraint %r pominięty: predykat nieznany "
                "w predicate_positions lub brakuje wymaganej roli.",
                tc.name,
            )
    return rules


# ---------------------------------------------------------------------------
# Implementacja wewnętrzna
# ---------------------------------------------------------------------------

_KEY_VAR        = "TcKey"
_EARLY_DATE_VAR = "TcEarlyDate"
_LATE_DATE_VAR  = "TcLateDate"


def _make_literal_args(
    roles: list[str],
    key_role: str,
    date_role: str,
    date_var: str,
) -> list[RuleArg] | None:
    """
    Buduje listę RuleArg dla literału LP z predykatu o danych rolach.

    Pierwsza rola pasująca do key_role  → VarTerm(_KEY_VAR)
    Pierwsza rola pasująca do date_role → VarTerm(date_var)
    Pozostałe role                       → VarTerm("_")

    Zwraca None jeśli key_role lub date_role nie znaleziono w roles.
    """
    args: list[RuleArg] = []
    found_key = False
    found_date = False

    for r in roles:
        ru = r.upper()
        if not found_key and ru == key_role.upper():
            args.append(RuleArg(role=r, term=VarTerm(var=_KEY_VAR)))
            found_key = True
        elif not found_date and ru == date_role.upper():
            args.append(RuleArg(role=r, term=VarTerm(var=date_var)))
            found_date = True
        else:
            args.append(RuleArg(role=r, term=VarTerm(var="_")))

    if not found_key or not found_date:
        return None
    return args


def _tc_to_rule(
    tc: TemporalConstraint,
    predicate_positions: dict[str, list[str]],
) -> Rule | None:
    """Konwertuje jeden TemporalConstraint do Rule, lub None jeśli niemożliwe."""
    earlier_roles = predicate_positions.get(tc.earlier_pred.lower())
    later_roles   = predicate_positions.get(tc.later_pred.lower())

    if earlier_roles is None:
        log.debug("TemporalConstraint %r: predykat %r nieznany.", tc.name, tc.earlier_pred)
        return None
    if later_roles is None:
        log.debug("TemporalConstraint %r: predykat %r nieznany.", tc.name, tc.later_pred)
        return None

    early_args = _make_literal_args(
        earlier_roles, tc.earlier_key_role, tc.earlier_date_role, _EARLY_DATE_VAR
    )
    if early_args is None:
        log.debug(
            "TemporalConstraint %r: rola klucza %r lub daty %r nie znaleziona w %r.",
            tc.name, tc.earlier_key_role, tc.earlier_date_role, tc.earlier_pred,
        )
        return None

    late_args = _make_literal_args(
        later_roles, tc.later_key_role, tc.later_date_role, _LATE_DATE_VAR
    )
    if late_args is None:
        log.debug(
            "TemporalConstraint %r: rola klucza %r lub daty %r nie znaleziona w %r.",
            tc.name, tc.later_key_role, tc.later_date_role, tc.later_pred,
        )
        return None

    return Rule(
        rule_id=f"tc_{tc.name}",
        language=RuleLanguage.horn_naf_stratified,
        head=RuleHead(
            predicate=tc.violation_atom,
            args=[
                RuleArg(role="CONSTRAINT", term=ConstTerm(const=tc.name)),
                RuleArg(role="KEY",        term=VarTerm(var=_KEY_VAR)),
            ],
        ),
        body=[
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate=tc.earlier_pred.lower(),
                args=early_args,
            ),
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate=tc.later_pred.lower(),
                args=late_args,
            ),
            # Naruszenie: późniejsza data jest ŚCIŚLE PRZED wcześniejszą datą.
            # Ten sam czas (D == D) → before(D, D) = False → brak naruszenia.
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate="before",
                args=[
                    RuleArg(role="FROM", term=VarTerm(var=_LATE_DATE_VAR)),
                    RuleArg(role="TO",   term=VarTerm(var=_EARLY_DATE_VAR)),
                ],
            ),
        ],
        metadata=RuleMetadata(stratum=0, learned=False),
    )


_DATE_A_VAR = "TcDateA"
_DATE_B_VAR = "TcDateB"


def _tcc_to_rule(
    tc: TemporalCoincidenceConstraint,
    predicate_positions: dict[str, list[str]],
) -> Rule | None:
    """Konwertuje TemporalCoincidenceConstraint do Rule z NAF (stratum=1)."""
    roles_a = predicate_positions.get(tc.pred_a.lower())
    roles_b = predicate_positions.get(tc.pred_b.lower())

    if roles_a is None:
        log.debug("TemporalCoincidenceConstraint %r: predykat %r nieznany.", tc.name, tc.pred_a)
        return None
    if roles_b is None:
        log.debug("TemporalCoincidenceConstraint %r: predykat %r nieznany.", tc.name, tc.pred_b)
        return None

    args_a = _make_literal_args(roles_a, tc.key_role_a, tc.date_role_a, _DATE_A_VAR)
    if args_a is None:
        log.debug(
            "TemporalCoincidenceConstraint %r: rola klucza %r lub daty %r nie znaleziona w %r.",
            tc.name, tc.key_role_a, tc.date_role_a, tc.pred_a,
        )
        return None

    args_b = _make_literal_args(roles_b, tc.key_role_b, tc.date_role_b, _DATE_B_VAR)
    if args_b is None:
        log.debug(
            "TemporalCoincidenceConstraint %r: rola klucza %r lub daty %r nie znaleziona w %r.",
            tc.name, tc.key_role_b, tc.date_role_b, tc.pred_b,
        )
        return None

    same_pred = _PERIOD_PREDICATE[tc.period]

    return Rule(
        rule_id=f"tcc_{tc.name}",
        language=RuleLanguage.horn_naf_stratified,
        head=RuleHead(
            predicate=tc.violation_atom,
            args=[
                RuleArg(role="CONSTRAINT", term=ConstTerm(const=tc.name)),
                RuleArg(role="KEY",        term=VarTerm(var=_KEY_VAR)),
            ],
        ),
        body=[
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate=tc.pred_a.lower(),
                args=args_a,
            ),
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate=tc.pred_b.lower(),
                args=args_b,
            ),
            # Naruszenie gdy NOT same_<period>(DA, DB).
            # same_* to computed base facts — NAF jest safe (stratum validator je pomija).
            RuleBodyLiteral(
                literal_type=LiteralType.naf,
                predicate=same_pred,
                args=[
                    RuleArg(role="FROM", term=VarTerm(var=_DATE_A_VAR)),
                    RuleArg(role="TO",   term=VarTerm(var=_DATE_B_VAR)),
                ],
            ),
        ],
        metadata=RuleMetadata(stratum=1, learned=False),
    )


_EARLY_VAR = "TcAnchorDate"
_LATER_VAR  = "TcEventDate"


def _tw_to_rule(
    tc: TemporalWindowConstraint,
    predicate_positions: dict[str, list[str]],
) -> Rule | None:
    """Konwertuje TemporalWindowConstraint do Rule z NAF (stratum=1)."""
    earlier_roles = predicate_positions.get(tc.earlier_pred.lower())
    later_roles   = predicate_positions.get(tc.later_pred.lower())

    if earlier_roles is None:
        log.debug("TemporalWindowConstraint %r: predykat %r nieznany.", tc.name, tc.earlier_pred)
        return None
    if later_roles is None:
        log.debug("TemporalWindowConstraint %r: predykat %r nieznany.", tc.name, tc.later_pred)
        return None

    early_args = _make_literal_args(earlier_roles, tc.earlier_key_role, tc.earlier_date_role, _EARLY_VAR)
    if early_args is None:
        log.debug(
            "TemporalWindowConstraint %r: rola klucza %r lub daty %r nie znaleziona w %r.",
            tc.name, tc.earlier_key_role, tc.earlier_date_role, tc.earlier_pred,
        )
        return None

    later_args = _make_literal_args(later_roles, tc.later_key_role, tc.later_date_role, _LATER_VAR)
    if later_args is None:
        log.debug(
            "TemporalWindowConstraint %r: rola klucza %r lub daty %r nie znaleziona w %r.",
            tc.name, tc.later_key_role, tc.later_date_role, tc.later_pred,
        )
        return None

    within_pred = tc.window_predicate()

    return Rule(
        rule_id=f"tw_{tc.name}",
        language=RuleLanguage.horn_naf_stratified,
        head=RuleHead(
            predicate=tc.violation_atom,
            args=[
                RuleArg(role="CONSTRAINT", term=ConstTerm(const=tc.name)),
                RuleArg(role="KEY",        term=VarTerm(var=_KEY_VAR)),
            ],
        ),
        body=[
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate=tc.earlier_pred.lower(),
                args=early_args,
            ),
            RuleBodyLiteral(
                literal_type=LiteralType.pos,
                predicate=tc.later_pred.lower(),
                args=later_args,
            ),
            # Naruszenie gdy DB NIE jest w oknie [DA, DA+n_days].
            # within_{n}_days_after to computed base fact — NAF jest safe.
            RuleBodyLiteral(
                literal_type=LiteralType.naf,
                predicate=within_pred,
                args=[
                    RuleArg(role="FROM", term=VarTerm(var=_EARLY_VAR)),
                    RuleArg(role="TO",   term=VarTerm(var=_LATER_VAR)),
                ],
            ),
        ],
        metadata=RuleMetadata(stratum=1, learned=False),
    )
