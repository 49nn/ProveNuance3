"""
SymbolicVerifier — fasada Symbolic Verifier.

Przepływ:
  1. Konwersja Fact + ClusterStateRow → LP strings (fakty bazowe Clingo)
  2. Budowa programu LP (fakty + reguły)
  3. Clingo.solve() → model stabilny (frozenset[Symbol])
  4. Symbol → GroundAtom (cały model)
  5. Ekstrakcja proof DAG (backtracking grounder na modelu)
  6. Aktualizacja statusów faktów: inferred_candidate → proved
  7. Tworzenie nowych Fact dla atomów derywowanych nieobecnych w input
"""
from __future__ import annotations

import uuid

from data_model.common import RoleArg, TruthDistribution
from data_model.fact import Fact, FactProvenance, FactStatus
from data_model.rule import Rule
from nn.graph_builder import ClusterSchema, ClusterStateRow
from sv.converter import IdRegistry, cluster_to_lp, fact_to_lp, symbol_to_atom
from sv.proof import ProofRun, build_proof_run, extract_proof_dag, ground_rule
from sv.runner import build_program, solve
from sv.schema import DEFAULT_CLUSTER_ROLES, PREDICATE_POSITIONS
from sv.stratification import validate_stratification
from sv.types import GroundAtom, VerifyResult


# Statusy niepodlegające nadpisaniu przez verifier (jak w nn/inference.py)
_KEEP_STATUS = {
    FactStatus.observed,
    FactStatus.proved,
    FactStatus.rejected,
    FactStatus.retracted,
}


class SymbolicVerifier:
    """
    Weryfikator symboliczny Horn+NAF ze stratyfikowaną negacją.

    Parametry:
      cluster_schemas:      lista ClusterSchema (do odczytu domain i top-1 wartości)
      predicate_positions:  kolejność ról pozycyjnych per predykat (z schema.py)
      cluster_roles:        (entity_role, value_role) per klaster (z schema.py)
      truth_threshold:      minimalny próg confidence dla faktów T wchodzących do LP
    """

    def __init__(
        self,
        cluster_schemas: list[ClusterSchema],
        predicate_positions: dict[str, list[str]] | None = None,
        cluster_roles: dict[str, tuple[str, str]] | None = None,
        truth_threshold: float = 0.5,
    ) -> None:
        self.cluster_schemas = {s.name: s for s in cluster_schemas}
        self.predicate_positions = predicate_positions or PREDICATE_POSITIONS
        self.cluster_roles = cluster_roles or DEFAULT_CLUSTER_ROLES
        self.truth_threshold = truth_threshold

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    @staticmethod
    def _derived_positions(rules: list[Rule]) -> dict[str, list[str]]:
        """
        Buduje mapowanie predykat → lista_ról dla głów reguł (predykaty derywowane).
        Np. 'contract_formed' → ['ORDER'], 'can_withdraw' → ['CUSTOMER', 'ORDER'].
        """
        result: dict[str, list[str]] = {}
        for rule in rules:
            pred = rule.head.predicate
            if pred not in result:
                result[pred] = [arg.role.upper() for arg in rule.head.args]
        return result

    def verify(
        self,
        facts: list[Fact],
        rules: list[Rule],
        cluster_states: list[ClusterStateRow],
    ) -> VerifyResult:
        """
        Przeprowadza weryfikację symboliczną.

        Zwraca VerifyResult z:
          - updated_facts: wejściowe fakty ze zaktualizowanymi statusami
          - new_facts:     nowe fakty derywowane przez reguły
          - derived_atoms: cały model stabilny (GroundAtom)
          - proof_nodes:   proof DAG
        """
        registry = IdRegistry()

        # Połączone mapowanie ról: bazowe + derywowane z głów reguł
        all_positions = {**self.predicate_positions, **self._derived_positions(rules)}

        # 0. Fail-fast: negacja musi być stratyfikowana.
        validate_stratification(rules)

        # 1. Konwersja do LP strings (all_positions dla observed/proved/inferred_candidate)
        fact_lp_list, atom_to_fact_id = self._facts_to_lp(facts, registry, all_positions)
        cluster_lp_list = self._clusters_to_lp(cluster_states, registry)
        base_lp = fact_lp_list + cluster_lp_list

        # 2. Zbiór bazowych atomów (tylko observed/proved — do proweniencji LP)
        base_atoms: set[GroundAtom] = {
            atom for atom, fid in atom_to_fact_id.items()
            if any(f.fact_id == fid and f.status in _KEEP_STATUS for f in facts)
        }

        # 3. Clingo → model stabilny
        program = build_program(rules, base_lp)
        symbols = solve(program)

        # 4. Symbol → GroundAtom (używamy rozszerzonego all_positions)
        derived: frozenset[GroundAtom] = frozenset(
            symbol_to_atom(s, all_positions, self.cluster_roles)
            for s in symbols
        )

        # 5. Proof DAG
        proof_nodes = extract_proof_dag(
            set(derived),
            base_atoms,
            rules,
            registry.mapping(),
            all_positions,
        )

        # 6. Aktualizacja statusów wejściowych faktów
        updated_facts = self._update_statuses(facts, derived, proof_nodes, atom_to_fact_id)

        # 7. Nowe fakty derywowane (nieobecne w input)
        new_facts = self._make_new_facts(
            derived - base_atoms, atom_to_fact_id, proof_nodes,
            registry.mapping(), all_positions,
        )

        return VerifyResult(
            updated_facts=updated_facts,
            new_facts=new_facts,
            derived_atoms=derived,
            proof_nodes=proof_nodes,
        )

    def build_proof_run(
        self,
        result: VerifyResult,
        query_atoms: list[GroundAtom],
        rules: list[Rule],
    ) -> ProofRun:
        """
        Buduje serializowalny ProofRun z wyników verify().
        Oddzielona metoda — nie zawsze potrzeba pełnego ProofRun.
        """
        all_positions = {**self.predicate_positions, **self._derived_positions(rules)}
        registry = IdRegistry()
        atom_to_fact_id: dict[GroundAtom, str] = {}
        for fact in (result.updated_facts + result.new_facts):
            atom = self._fact_to_ground_atom(fact, registry, all_positions)
            if atom is not None:
                atom_to_fact_id[atom] = fact.fact_id

        rules_index = {r.rule_id: r for r in rules}
        return build_proof_run(
            result.proof_nodes,
            query_atoms,
            atom_to_fact_id,
            id_map=registry.mapping(),
            rules_index=rules_index,
        )

    def classify_query_atom(
        self,
        query_atom: GroundAtom,
        result: VerifyResult,
        rules: list[Rule],
    ) -> str:
        """
        Klasyfikuje wynik zapytania:
          - proved: atom jest w modelu
          - blocked: istnieje ground reguła dla celu, ale NAF jest naruszone
          - not_proved: istnieją reguły dla celu, ale nie udało się wyprowadzić
          - unknown: brak reguł i brak atomu w modelu
        """
        model = set(result.derived_atoms)
        all_positions = {**self.predicate_positions, **self._derived_positions(rules)}
        if query_atom in model:
            return "proved"

        has_rule_for_head = False
        blocked = False
        for rule in rules:
            if rule.head.predicate != query_atom.predicate:
                continue
            has_rule_for_head = True
            for grounded in ground_rule(rule, model, all_positions):
                if grounded.head != query_atom:
                    continue
                if not all(a in model for a in grounded.pos_body):
                    continue
                if any(self._naf_violated(naf_atom, result.derived_atoms) for naf_atom in grounded.neg_body):
                    blocked = True
                    break
            if blocked:
                break

        if blocked:
            return "blocked"
        if has_rule_for_head:
            return "not_proved"
        return "unknown"

    # ------------------------------------------------------------------
    # Prywatne metody pomocnicze
    # ------------------------------------------------------------------

    def _facts_to_lp(
        self,
        facts: list[Fact],
        registry: IdRegistry,
        all_positions: dict[str, list[str]],
    ) -> tuple[list[str], dict[GroundAtom, str]]:
        """
        Konwertuje listę faktów do LP strings + słownik atom→fact_id.

        Do programu LP (lp_lines) trafiają tylko fakty z statusem observed/proved
        — inferred_candidate nie jest asercją, nie powinny być bazą LP.
        Do atom_to_fact_id trafiają WSZYSTKIE fakty T (dla aktualizacji statusów).
        """
        lp_lines: list[str] = []
        atom_to_fact_id: dict[GroundAtom, str] = {}

        for fact in facts:
            if fact.truth.value != "T":
                continue
            if fact.truth.confidence is not None and fact.truth.confidence < self.truth_threshold:
                continue
            atom = self._fact_to_ground_atom(fact, registry, all_positions)
            if atom is None:
                continue
            atom_to_fact_id[atom] = fact.fact_id
            # LP string tylko dla pewnych faktów (nie neural candidates)
            if fact.status in _KEEP_STATUS:
                lp = fact_to_lp(fact, registry, all_positions, self.truth_threshold)
                if lp is not None:
                    lp_lines.append(lp)

        return lp_lines, atom_to_fact_id

    def _fact_to_ground_atom(
        self, fact: Fact, registry: IdRegistry,
        all_positions: dict[str, list[str]] | None = None,
    ) -> GroundAtom | None:
        """Konwertuje Fact → GroundAtom (role-based, clingo IDs)."""
        pred = fact.predicate.lower()
        positions = all_positions or self.predicate_positions
        roles = positions.get(pred)
        if roles is None:
            return None
        role_map: dict[str, str] = {
            arg.role.upper(): arg.entity_id or arg.literal_value  # type: ignore[arg-type]
            for arg in fact.args
        }
        bindings = tuple(sorted(
            (r, registry.register(role_map[r]))
            for r in roles
            if r in role_map
        ))
        if len(bindings) != len(roles):
            return None
        return GroundAtom(pred, bindings)

    def _clusters_to_lp(
        self,
        states: list[ClusterStateRow],
        registry: IdRegistry,
    ) -> list[str]:
        """Konwertuje ClusterStateRow do LP strings (top-1 wartość per klaster)."""
        lines: list[str] = []
        for state in states:
            schema = self.cluster_schemas.get(state.cluster_name)
            if schema is None:
                continue
            lp = cluster_to_lp(state, schema, registry, self.cluster_roles)
            if lp is not None:
                lines.append(lp)
        return lines

    def _update_statuses(
        self,
        facts: list[Fact],
        derived: frozenset[GroundAtom],
        proof_nodes: dict[GroundAtom, ...],
        atom_to_fact_id: dict[GroundAtom, str],
    ) -> list[Fact]:
        """
        Aktualizuje statusy faktów:
          - inferred_candidate + atom w derived + reguła dowodu   → proved (+proof_id)
          - inferred_candidate + atom w derived (fakt bazowy)     → proved
          - inferred_candidate + brak atomu w modelu              → inferred_candidate
          - observed / proved / rejected / retracted              → bez zmian
        Używa model_copy() (Pydantic immutability).
        """
        fact_id_to_atom: dict[str, GroundAtom] = {v: k for k, v in atom_to_fact_id.items()}
        updated: list[Fact] = []

        for fact in facts:
            if fact.status in _KEEP_STATUS:
                updated.append(fact)
                continue

            atom = fact_id_to_atom.get(fact.fact_id)
            if atom is not None and atom in derived:
                node = proof_nodes.get(atom)
                if node is not None and node.rule_id is not None:  # derywowany przez regułę
                    provenance = (fact.provenance or FactProvenance()).model_copy(
                        update={"proof_id": node.rule_id}
                    )
                    updated.append(fact.model_copy(update={
                        "status":     FactStatus.proved,
                        "provenance": provenance,
                    }))
                else:
                    # Faktu nie wyprowadziła reguła, ale jest obecny w modelu
                    # (np. jako ekstensionalny atom bazowy).
                    updated.append(fact.model_copy(update={"status": FactStatus.proved}))
                continue

            updated.append(fact)

        return updated

    @staticmethod
    def _naf_violated(
        naf_atom: GroundAtom,
        derived: frozenset[GroundAtom],
    ) -> bool:
        """
        True gdy istnieje atom w modelu pasujący do częściowego atomu NAF.
        """
        required = set(naf_atom.bindings)
        for atom in derived:
            if atom.predicate != naf_atom.predicate:
                continue
            if required.issubset(atom.bindings):
                return True
        return False

    def _make_new_facts(
        self,
        new_atoms: frozenset[GroundAtom],
        existing_atom_to_fact_id: dict[GroundAtom, str],
        proof_nodes: dict[GroundAtom, ...],
        id_map: dict[str, str],
        all_positions: dict[str, list[str]] | None = None,
    ) -> list[Fact]:
        """
        Tworzy Fact obiekty dla atomów derywowanych nieobecnych w input.

        Tworzone są Fact dla:
          - znanych predykatów bazowych (z predicate_positions)
          - konkluzji derywowanych przez reguły (np. contract_formed, can_withdraw)
        Pomijane są:
          - predykaty wewnętrzne (_sv_*)
          - predykaty wyjątków (ab_*)
          - predykaty pomocnicze (prepaid, returned_or_proof, responded_in_14_days, ...)
            które są intermediary, nie końcowymi konkluzjami
        """
        # Predykaty pomocnicze — nie eksponujemy ich jako Fact
        _SKIP_PREFIXES = ("_sv_", "ab_")
        _SKIP_EXACT = {
            "prepaid", "returned_or_proof", "responded_in_14_days",
            "paid_within_48h", "coupon_not_expired", "meets_min_basket",
            "within_14_days", "order_mentioned",
        }

        new_facts: list[Fact] = []

        for atom in new_atoms:
            if atom in existing_atom_to_fact_id:
                continue
            if any(atom.predicate.startswith(p) for p in _SKIP_PREFIXES):
                continue
            if atom.predicate in _SKIP_EXACT:
                continue

            node = proof_nodes.get(atom)
            if node is None or node.rule_id is None:
                continue  # niesuperweniowany atom — pomijamy

            proof_id = node.rule_id

            # Budujemy RoleArg z bindingów GroundAtom
            # (rola → entity_id; oryginalne ID przez id_map)
            args: list[RoleArg] = [
                RoleArg(role=role, entity_id=id_map.get(val, val))
                for role, val in atom.bindings
            ]

            new_facts.append(Fact(
                fact_id=str(uuid.uuid4()),
                predicate=atom.predicate.upper(),
                arity=len(args),
                args=args,
                truth=TruthDistribution(domain=["T", "F", "U"], value="T", confidence=1.0),
                status=FactStatus.proved,
                provenance=FactProvenance(proof_id=proof_id),
            ))

        return new_facts
