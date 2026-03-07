CREATE TABLE IF NOT EXISTS proof_candidate_feedback (
    id                  SERIAL PRIMARY KEY,
    proof_id            TEXT NOT NULL REFERENCES proof_runs(proof_id) ON DELETE CASCADE,
    fact_id             TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    predicate           TEXT NOT NULL,
    outcome             TEXT NOT NULL CHECK (outcome IN ('proved', 'blocked', 'not_proved', 'unknown')),
    atom_text           TEXT,
    violated_naf        TEXT[] NOT NULL DEFAULT '{}',
    missing_pos_body    TEXT[] NOT NULL DEFAULT '{}',
    supporting_rule_ids TEXT[] NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (proof_id, fact_id)
);

CREATE INDEX IF NOT EXISTS proof_candidate_feedback_proof_idx
    ON proof_candidate_feedback (proof_id, outcome);

CREATE INDEX IF NOT EXISTS proof_candidate_feedback_fact_idx
    ON proof_candidate_feedback (fact_id);
