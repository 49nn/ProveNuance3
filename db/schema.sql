-- ProveNuance3 – PostgreSQL schema
-- Derived from: schemas/common.json, schemas/entity.json,
--               schemas/fact.json, schemas/rule.json
-- Requires: pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- TYPY WYLICZENIOWE
-- ============================================================

CREATE TYPE truth_value      AS ENUM ('T', 'F', 'U');
CREATE TYPE fact_status      AS ENUM ('observed', 'inferred_candidate', 'proved', 'rejected', 'retracted');
CREATE TYPE literal_type     AS ENUM ('pos', 'naf');
CREATE TYPE rule_language    AS ENUM ('horn_naf_stratified');
CREATE TYPE clamp_source_t   AS ENUM ('text', 'memory', 'manual');

-- ============================================================
-- ONTOLOGIA
-- ============================================================

CREATE TABLE entity_types (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    source_span_text TEXT
);

CREATE TABLE predicate_definitions (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    source_span_text TEXT
);

-- Pozycjonowane role predykatu; entity_type_id=NULL oznacza literał (DATE, AMOUNT, RESULT…)
CREATE TABLE predicate_roles (
    predicate_id    INTEGER NOT NULL REFERENCES predicate_definitions(id) ON DELETE CASCADE,
    position        INTEGER NOT NULL,
    role_name       TEXT    NOT NULL,
    entity_type_id  INTEGER REFERENCES entity_types(id),
    PRIMARY KEY (predicate_id, position)
);

-- Klastry unarne: zmienne dyskretne z softmax
CREATE TABLE cluster_definitions (
    id              SERIAL PRIMARY KEY,
    name            TEXT    NOT NULL UNIQUE,
    entity_type_id  INTEGER NOT NULL REFERENCES entity_types(id),
    entity_role_name TEXT   NOT NULL,
    value_role_name  TEXT   NOT NULL,
    description     TEXT,
    source_span_text TEXT
);

CREATE TABLE cluster_domain_values (
    cluster_id  INTEGER NOT NULL REFERENCES cluster_definitions(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    value       TEXT    NOT NULL,
    PRIMARY KEY (cluster_id, position)
);

-- ============================================================
-- ŹRÓDŁA (dokumenty, teksty – podstawa provenance)
-- Odpowiada ProvenanceItem.source_id z common.json
-- ============================================================

CREATE TABLE sources (
    id          SERIAL PRIMARY KEY,
    source_id   TEXT NOT NULL UNIQUE,   -- klucz biznesowy używany w provenance
    title       TEXT,
    content     TEXT,                   -- surowy tekst
    source_type TEXT,                   -- 'case_text' | 'document' | 'manual'
    source_rank INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- CASE'Y TESTOWE (nakładka na sources)
-- ============================================================

CREATE TABLE cases (
    id          SERIAL PRIMARY KEY,
    case_id     TEXT    NOT NULL UNIQUE,
    source_id   INTEGER NOT NULL REFERENCES sources(id),
    title       TEXT,
    dataset_split TEXT CHECK (dataset_split IN ('train_gold', 'train_unlabeled', 'holdout')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE case_queries (
    id              SERIAL PRIMARY KEY,
    case_id         INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    query           TEXT    NOT NULL,
    expected_result TEXT    CHECK (expected_result IN ('proved','not_proved','blocked','unknown')),
    notes           TEXT
);

-- ============================================================
-- SELF-TRAINING ROUNDS AND PSEUDO-LABELS
-- ============================================================

CREATE TABLE self_training_rounds (
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

CREATE TABLE pseudo_fact_labels (
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

CREATE INDEX pseudo_fact_labels_case_idx
    ON pseudo_fact_labels (case_id);

CREATE INDEX pseudo_fact_labels_round_idx
    ON pseudo_fact_labels (round_id, accepted, promoted);

CREATE TABLE pseudo_cluster_labels (
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

CREATE INDEX pseudo_cluster_labels_case_idx
    ON pseudo_cluster_labels (case_id);

CREATE INDEX pseudo_cluster_labels_round_idx
    ON pseudo_cluster_labels (round_id, accepted, promoted);

-- ============================================================
-- ENCJE (entity.json)
-- ============================================================

CREATE TABLE entities (
    id              SERIAL PRIMARY KEY,
    entity_id       TEXT    NOT NULL UNIQUE,    -- 'O100', 'CUST1', '2026-02-01', …
    entity_type_id  INTEGER NOT NULL REFERENCES entity_types(id),
    canonical_name  TEXT    NOT NULL,           -- wymagane przez entity.json
    embedding       vector(768),                -- pgvector; dim zależny od modelu
    embedding_ref   TEXT,                       -- opcjonalny external ref (S3, FS)
    blocking_keys   TEXT[],                     -- entity.json → linking.blocking_keys
    last_linked_from TEXT[],                    -- entity.json → linking.last_linked_from
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX entities_embedding_idx
    ON entities USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE entity_aliases (
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias       TEXT    NOT NULL,
    PRIMARY KEY (entity_id, alias)
);

-- entity.json → provenance[] – tablica ProvenanceItem na poziomie encji
CREATE TABLE entity_provenance (
    id          SERIAL PRIMARY KEY,
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    source_id   TEXT    NOT NULL,           -- → sources.source_id
    span_start  INTEGER,                    -- common.json → Span.start (pojedynczy span)
    span_end    INTEGER,                    -- common.json → Span.end  (pojedynczy span)
    spans       JSONB,                      -- array [{start,end}] gdy wiele spanów
    extractor   TEXT,
    confidence  DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    note        TEXT,
    -- common.json → ProvenanceItem: span i spans wzajemnie się wykluczają
    CHECK (NOT (span_start IS NOT NULL AND spans IS NOT NULL)),
    -- span_start i span_end muszą być ustawione razem lub wcale
    CHECK ((span_start IS NULL) = (span_end IS NULL))
);

-- entity.json → memory_slots – wersjonowane wartości instancyjne
CREATE TABLE entity_slots (
    id          SERIAL PRIMARY KEY,
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    slot_name   TEXT    NOT NULL,
    value       JSONB   NOT NULL,           -- dowolna wartość (string, number, object)
    normalized  JSONB,                      -- opcjonalna znormalizowana forma
    valid_from  TIMESTAMPTZ,
    valid_to    TIMESTAMPTZ,
    confidence  DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    source_rank DOUBLE PRECISION CHECK (source_rank BETWEEN 0 AND 1),
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX entity_slots_lookup
    ON entity_slots (entity_id, slot_name, version DESC);

-- entity.json → memory_slots[*].provenance[] – ProvenanceItem na poziomie slotu
CREATE TABLE entity_slot_provenance (
    id          SERIAL PRIMARY KEY,
    slot_id     INTEGER NOT NULL REFERENCES entity_slots(id) ON DELETE CASCADE,
    source_id   TEXT    NOT NULL,
    span_start  INTEGER,
    span_end    INTEGER,
    spans       JSONB,
    extractor   TEXT,
    confidence  DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    note        TEXT,
    CHECK (NOT (span_start IS NOT NULL AND spans IS NOT NULL)),
    CHECK ((span_start IS NULL) = (span_end IS NULL))
);

-- ============================================================
-- FAKTY REIFIKOWANE (fact.json)
-- ============================================================

CREATE TABLE facts (
    id                  SERIAL PRIMARY KEY,
    fact_id             TEXT        NOT NULL UNIQUE,    -- fact.json → fact_id
    predicate           TEXT        NOT NULL,            -- fact.json → predicate
    arity               INTEGER,                         -- fact.json → arity
    status              fact_status NOT NULL,            -- fact.json → status

    -- fact.json → truth (TruthDistribution z common.json)
    truth_domain        TEXT[]      NOT NULL DEFAULT '{T,F,U}',
    truth_value         truth_value,
    truth_confidence    DOUBLE PRECISION CHECK (truth_confidence BETWEEN 0 AND 1),
    truth_logits        JSONB,      -- {"T": 1.2, "F": -0.5, "U": 0.1}
    -- NOTE: klucze tylko T/F/U – walidacja przez aplikację (PostgreSQL nie obsługuje subquery w CHECK)

    -- fact.json → time
    event_time          TIMESTAMPTZ,
    valid_from          TIMESTAMPTZ,
    valid_to            TIMESTAMPTZ,

    -- fact.json → source
    source_id           TEXT,       -- → sources.source_id
    source_spans        JSONB,      -- array of Span [{start,end}]
    source_extractor    TEXT,
    source_confidence   DOUBLE PRECISION CHECK (source_confidence BETWEEN 0 AND 1),

    -- fact.json → provenance.proof_id
    proof_id            TEXT,       -- → proof_runs.proof_id

    -- fact.json → constraints_tags
    constraints_tags    TEXT[],

    case_id             INTEGER REFERENCES cases(id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX facts_predicate_idx    ON facts (predicate);
CREATE INDEX facts_status_idx       ON facts (status);
CREATE INDEX facts_case_idx         ON facts (case_id);

-- fact.json → args[] – RoleArg z common.json
-- Dokładnie jedno z: entity_id (referencja) lub literal_value (wartość inline)
CREATE TABLE fact_args (
    fact_id       INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    role          TEXT    NOT NULL,         -- common.json → RoleArg.role
    entity_id     TEXT,                     -- common.json → RoleArg.entity_id
    literal_value TEXT,                     -- common.json → RoleArg.literal_value
    type_hint     TEXT,                     -- common.json → RoleArg.type_hint
    PRIMARY KEY (fact_id, position),
    -- common.json → RoleArg oneOf: dokładnie jedno z entity_id / literal_value
    CHECK ((entity_id IS NOT NULL) != (literal_value IS NOT NULL))
);

CREATE INDEX fact_args_entity_idx ON fact_args (entity_id);

-- fact.json → provenance.neural_trace[]
CREATE TABLE fact_neural_trace (
    id              SERIAL PRIMARY KEY,
    fact_id         INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    step            INTEGER NOT NULL,
    edge_type       TEXT    NOT NULL,
    from_fact_id    TEXT,                       -- → facts.fact_id
    from_cluster_id TEXT,                       -- → cluster_definitions.name
    delta_logits    JSONB   NOT NULL,           -- {"T": 0.3, "F": -0.1, "U": 0.05}
    -- fact.json → neural_trace oneOf: dokładnie jedno źródło wkładu
    CHECK ((from_fact_id IS NOT NULL) != (from_cluster_id IS NOT NULL))
    -- NOTE: delta_logits klucze tylko T/F/U – walidacja przez aplikację
);

-- ============================================================
-- REGUŁY HORN + NAF (rule.json)
-- ============================================================

CREATE TABLE rule_modules (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE rules (
    id          SERIAL PRIMARY KEY,
    rule_id     TEXT        NOT NULL UNIQUE,    -- rule.json → rule_id
    module_id   INTEGER     NOT NULL REFERENCES rule_modules(id) ON DELETE CASCADE,
    language    rule_language NOT NULL DEFAULT 'horn_naf_stratified',

    -- rule.json → head: {predicate, args: [RuleArg]}
    head        JSONB       NOT NULL,
    -- rule.json → body: [{literal_type, predicate, args: [RuleArg]}]
    body        JSONB       NOT NULL,

    -- skompilowana forma dla Clingo (opcjonalna, generowana z head/body)
    clingo_text TEXT,

    -- rule.json → metadata
    stratum         INTEGER         NOT NULL DEFAULT 0,
    learned         BOOLEAN         NOT NULL DEFAULT FALSE,
    weight          DOUBLE PRECISION CHECK (weight >= 0),
    support         INTEGER         CHECK (support >= 0),
    precision_est   DOUBLE PRECISION CHECK (precision_est BETWEEN 0 AND 1),
    last_validated_at TIMESTAMPTZ,
    constraints     TEXT[],

    enabled     BOOLEAN     NOT NULL DEFAULT TRUE,
    version     INTEGER     NOT NULL DEFAULT 1,
    source_span_text TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- STANY KLASTRÓW (logity / softmax per encja × klaster × case)
-- ============================================================

CREATE TABLE cluster_states (
    id              SERIAL PRIMARY KEY,
    entity_id       INTEGER     NOT NULL REFERENCES entities(id)           ON DELETE CASCADE,
    cluster_id      INTEGER     NOT NULL REFERENCES cluster_definitions(id) ON DELETE CASCADE,
    case_id         INTEGER     REFERENCES cases(id),
    logits          FLOAT[]     NOT NULL,
    is_clamped      BOOLEAN     NOT NULL DEFAULT FALSE,
    clamp_hard      BOOLEAN     NOT NULL DEFAULT FALSE,
    clamp_source    clamp_source_t,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_id, cluster_id, case_id)
);

-- ============================================================
-- PROVENANCE – symboliczne dowody (proof_id referenced by facts.proof_id)
-- ============================================================

CREATE TABLE proof_runs (
    id          SERIAL PRIMARY KEY,
    proof_id    TEXT        NOT NULL UNIQUE,    -- klucz biznesowy referencjonowany przez facts
    case_id     INTEGER     NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    query       TEXT        NOT NULL,
    result      TEXT        NOT NULL CHECK (result IN ('proved','not_proved','unknown')),
    proof_dag   JSONB,      -- DAG kroków: [{rule_id, substitution, used_fact_ids}]
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Poszczególne kroki dowodu (dla odpytywań po faktach/regułach)
CREATE TABLE proof_steps (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES proof_runs(id) ON DELETE CASCADE,
    step_order      INTEGER NOT NULL,
    rule_id         TEXT,               -- → rules.rule_id
    rule_text       TEXT,               -- kopia clingo_text w momencie dowodu
    substitution    JSONB,              -- {"C": "CUST1", "O": "O100"}
    used_fact_ids   TEXT[]              -- array fact_id strings
);

CREATE TABLE proof_candidate_feedback (
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

CREATE INDEX proof_candidate_feedback_proof_idx
    ON proof_candidate_feedback (proof_id, outcome);

CREATE INDEX proof_candidate_feedback_fact_idx
    ON proof_candidate_feedback (fact_id);
