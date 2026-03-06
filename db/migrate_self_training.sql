ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS dataset_split TEXT;

ALTER TABLE cases
    DROP CONSTRAINT IF EXISTS cases_dataset_split_check;

ALTER TABLE cases
    ADD CONSTRAINT cases_dataset_split_check
    CHECK (dataset_split IN ('train_gold', 'train_unlabeled', 'holdout'));

CREATE TABLE IF NOT EXISTS self_training_rounds (
    id                          SERIAL PRIMARY KEY,
    round_id                    TEXT NOT NULL UNIQUE,
    parent_round_id             TEXT REFERENCES self_training_rounds(round_id),
    status                      TEXT NOT NULL DEFAULT 'draft'
                                CHECK (status IN ('draft', 'collected', 'imported', 'promoted', 'rejected')),
    teacher_module              TEXT,
    fact_conf_threshold         DOUBLE PRECISION NOT NULL CHECK (fact_conf_threshold BETWEEN 0 AND 1),
    cluster_top1_threshold      DOUBLE PRECISION NOT NULL CHECK (cluster_top1_threshold BETWEEN 0 AND 1),
    cluster_margin_threshold    DOUBLE PRECISION NOT NULL CHECK (cluster_margin_threshold BETWEEN 0 AND 1),
    notes                       TEXT,
    promoted_at                 TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pseudo_fact_labels (
    id                  SERIAL PRIMARY KEY,
    round_id            INTEGER NOT NULL REFERENCES self_training_rounds(id) ON DELETE CASCADE,
    case_id             INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    fact_key            TEXT NOT NULL,
    fact_json           JSONB NOT NULL,
    truth_confidence    DOUBLE PRECISION NOT NULL CHECK (truth_confidence BETWEEN 0 AND 1),
    proof_id            TEXT,
    accepted            BOOLEAN NOT NULL DEFAULT TRUE,
    rejection_reason    TEXT,
    promoted            BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (round_id, case_id, fact_key)
);

CREATE INDEX IF NOT EXISTS pseudo_fact_labels_case_idx
    ON pseudo_fact_labels (case_id);

CREATE INDEX IF NOT EXISTS pseudo_fact_labels_round_idx
    ON pseudo_fact_labels (round_id, accepted, promoted);

CREATE TABLE IF NOT EXISTS pseudo_cluster_labels (
    id                  SERIAL PRIMARY KEY,
    round_id            INTEGER NOT NULL REFERENCES self_training_rounds(id) ON DELETE CASCADE,
    case_id             INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    entity_id           TEXT NOT NULL,
    cluster_name        TEXT NOT NULL,
    value               TEXT NOT NULL,
    state_json          JSONB NOT NULL,
    top1_confidence     DOUBLE PRECISION NOT NULL CHECK (top1_confidence BETWEEN 0 AND 1),
    margin              DOUBLE PRECISION NOT NULL CHECK (margin BETWEEN 0 AND 1),
    accepted            BOOLEAN NOT NULL DEFAULT TRUE,
    rejection_reason    TEXT,
    promoted            BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (round_id, case_id, entity_id, cluster_name)
);

CREATE INDEX IF NOT EXISTS pseudo_cluster_labels_case_idx
    ON pseudo_cluster_labels (case_id);

CREATE INDEX IF NOT EXISTS pseudo_cluster_labels_round_idx
    ON pseudo_cluster_labels (round_id, accepted, promoted);
