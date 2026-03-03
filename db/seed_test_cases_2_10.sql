-- ProveNuance3 – seed przypadków testowych TC-002 … TC-010
-- Pokrywa przypadki 2–10 z docs/test use case.md §4

-- ============================================================
-- SOURCES
-- ============================================================

INSERT INTO sources (source_id, title, source_type, source_rank) VALUES
    ('TC-002-TEXT', 'Brak płatności 48h – przelew (O101)',               'case_text', 10),
    ('TC-003-TEXT', 'COD – brak podstawy do anulowania (O102)',          'case_text', 10),
    ('TC-004-TEXT', 'Odstąpienie konsumenta w terminie (O103)',          'case_text', 10),
    ('TC-005-TEXT', 'Odstąpienie po terminie 14 dni (O104)',             'case_text', 10),
    ('TC-006-TEXT', 'Wyjątek – produkt na zamówienie (O105)',            'case_text', 10),
    ('TC-007-TEXT', 'Wyjątek – cyfrowy + zgoda + pobieranie (O106)',     'case_text', 10),
    ('TC-008-TEXT', 'Refund hold – brak zwrotu i dowodu (O107)',         'case_text', 10),
    ('TC-009-TEXT', 'Reklamacja uznana – brak odpowiedzi sklep (C900)',  'case_text', 10),
    ('TC-010-TEXT', 'Kupony: SAVE10 (nie łączy), EXTRA5 (nieważny)',     'case_text', 10);

-- ============================================================
-- CASES
-- ============================================================

INSERT INTO cases (case_id, source_id, title)
SELECT v.case_id, s.id, v.title
FROM (VALUES
    ('TC-002', 'TC-002-TEXT', 'Brak płatności 48h – przelew (O101)'),
    ('TC-003', 'TC-003-TEXT', 'COD – brak podstawy do anulowania (O102)'),
    ('TC-004', 'TC-004-TEXT', 'Odstąpienie konsumenta – 10 dni (O103)'),
    ('TC-005', 'TC-005-TEXT', 'Odstąpienie po terminie – 19 dni (O104)'),
    ('TC-006', 'TC-006-TEXT', 'Wyjątek – produkt na zamówienie (O105)'),
    ('TC-007', 'TC-007-TEXT', 'Wyjątek – cyfrowy + zgoda + pobieranie (O106)'),
    ('TC-008', 'TC-008-TEXT', 'Refund hold – brak zwrotu i brak dowodu (O107)'),
    ('TC-009', 'TC-009-TEXT', 'Reklamacja C900 – brak odpowiedzi w 14 dniach'),
    ('TC-010', 'TC-010-TEXT', 'Kupony SAVE10 + EXTRA5 dla O109')
) AS v(case_id, source_id, title)
JOIN sources s ON s.source_id = v.source_id;

-- ============================================================
-- ENTITIES
-- ============================================================

INSERT INTO entities (entity_id, entity_type_id, canonical_name) VALUES
    -- TC-002
    ('CUST2',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Anna Nowak'),
    ('O101',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #101'),
    -- TC-003
    ('CUST3',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Piotr Wiśniewski'),
    ('O102',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #102'),
    -- TC-004
    ('CUST4',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Maria Kowalczyk'),
    ('O103',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #103'),
    ('DEL103',   (SELECT id FROM entity_types WHERE name='DELIVERY'),        'Dostawa DHL #DEL103'),
    ('STMT103',  (SELECT id FROM entity_types WHERE name='STATEMENT'),       'Oświadczenie odstąpienie O103'),
    -- TC-005
    ('CUST5',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Tomasz Wójcik'),
    ('O104',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #104'),
    ('DEL104',   (SELECT id FROM entity_types WHERE name='DELIVERY'),        'Dostawa DHL #DEL104'),
    ('STMT104',  (SELECT id FROM entity_types WHERE name='STATEMENT'),       'Oświadczenie odstąpienie O104'),
    -- TC-006
    ('CUST6',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Karolina Lewandowska'),
    ('O105',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #105 – meble na wymiar'),
    ('PROD105',  (SELECT id FROM entity_types WHERE name='PRODUCT'),         'Meble na wymiar'),
    ('DEL105',   (SELECT id FROM entity_types WHERE name='DELIVERY'),        'Dostawa DHL #DEL105'),
    ('STMT105',  (SELECT id FROM entity_types WHERE name='STATEMENT'),       'Oświadczenie odstąpienie O105'),
    -- TC-007
    ('CUST7',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Michał Zieliński'),
    ('O106',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #106 – produkt cyfrowy'),
    ('PROD106',  (SELECT id FROM entity_types WHERE name='PRODUCT'),         'E-book / licencja cyfrowa'),
    -- TC-008
    ('CUST8',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Agnieszka Szymańska'),
    ('O107',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #107'),
    ('DEL107',   (SELECT id FROM entity_types WHERE name='DELIVERY'),        'Dostawa DHL #DEL107'),
    ('STMT107',  (SELECT id FROM entity_types WHERE name='STATEMENT'),       'Oświadczenie odstąpienie O107'),
    -- TC-009
    ('CUST9',    (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Marek Jankowski'),
    ('O108',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #108'),
    ('COMP900',  (SELECT id FROM entity_types WHERE name='COMPLAINT'),       'Reklamacja C900'),
    -- TC-010
    ('CUST10',   (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Barbara Kamińska'),
    ('O109',     (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #109'),
    ('CPNSAVE10',(SELECT id FROM entity_types WHERE name='COUPON'),          'Kupon SAVE10 – nie łączy się z innymi'),
    ('CPNEXTRA5',(SELECT id FROM entity_types WHERE name='COUPON'),          'Kupon EXTRA5 – min. koszyk 200 zł');

-- ============================================================
-- FACTS + ARGS
-- ============================================================

-- TC-002: ORDER_PLACED + PAYMENT_SELECTED(transfer), brak PAYMENT_MADE
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-002-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-002')
    FROM (VALUES
        ('F201', 'ORDER_PLACED',     3, '2026-03-01 10:00:00'),
        ('F202', 'PAYMENT_SELECTED', 2, '2026-03-01 10:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F201', 0, 'CUSTOMER',       'CUST2', NULL),
    ('F201', 1, 'ORDER',          'O101',  NULL),
    ('F201', 2, 'DATE',           NULL,    '2026-03-01'),
    ('F202', 0, 'ORDER',          'O101',  NULL),
    ('F202', 1, 'PAYMENT_METHOD', NULL,    'transfer')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-003: ORDER_PLACED + PAYMENT_SELECTED(cod)
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-003-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-003')
    FROM (VALUES
        ('F301', 'ORDER_PLACED',     3, '2026-03-01 10:00:00'),
        ('F302', 'PAYMENT_SELECTED', 2, '2026-03-01 10:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F301', 0, 'CUSTOMER',       'CUST3', NULL),
    ('F301', 1, 'ORDER',          'O102',  NULL),
    ('F301', 2, 'DATE',           NULL,    '2026-03-01'),
    ('F302', 0, 'ORDER',          'O102',  NULL),
    ('F302', 1, 'PAYMENT_METHOD', NULL,    'cod')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-004: DELIVERED(2026-02-01) + WITHDRAWAL_STATEMENT_SUBMITTED(2026-02-11) → 10 dni, w terminie
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-004-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-004')
    FROM (VALUES
        ('F401', 'DELIVERED',                      3, '2026-02-01 14:00:00'),
        ('F402', 'WITHDRAWAL_STATEMENT_SUBMITTED', 4, '2026-02-11 10:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F401', 0, 'ORDER',     'O103',   NULL),
    ('F401', 1, 'DELIVERY',  'DEL103', NULL),
    ('F401', 2, 'DATE',      NULL,     '2026-02-01'),
    ('F402', 0, 'CUSTOMER',  'CUST4',  NULL),
    ('F402', 1, 'ORDER',     'O103',   NULL),
    ('F402', 2, 'STATEMENT', 'STMT103',NULL),
    ('F402', 3, 'DATE',      NULL,     '2026-02-11')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-005: DELIVERED(2026-02-01) + WITHDRAWAL_STATEMENT_SUBMITTED(2026-02-20) → 19 dni, po terminie
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-005-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-005')
    FROM (VALUES
        ('F501', 'DELIVERED',                      3, '2026-02-01 14:00:00'),
        ('F502', 'WITHDRAWAL_STATEMENT_SUBMITTED', 4, '2026-02-20 10:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F501', 0, 'ORDER',     'O104',   NULL),
    ('F501', 1, 'DELIVERY',  'DEL104', NULL),
    ('F501', 2, 'DATE',      NULL,     '2026-02-01'),
    ('F502', 0, 'CUSTOMER',  'CUST5',  NULL),
    ('F502', 1, 'ORDER',     'O104',   NULL),
    ('F502', 2, 'STATEMENT', 'STMT104',NULL),
    ('F502', 3, 'DATE',      NULL,     '2026-02-20')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-006: DELIVERED(2026-02-01) + WITHDRAWAL_STATEMENT_SUBMITTED(2026-02-05), produkt CUSTOM
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-006-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-006')
    FROM (VALUES
        ('F601', 'DELIVERED',                      3, '2026-02-01 14:00:00'),
        ('F602', 'WITHDRAWAL_STATEMENT_SUBMITTED', 4, '2026-02-05 10:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F601', 0, 'ORDER',     'O105',   NULL),
    ('F601', 1, 'DELIVERY',  'DEL105', NULL),
    ('F601', 2, 'DATE',      NULL,     '2026-02-01'),
    ('F602', 0, 'CUSTOMER',  'CUST6',  NULL),
    ('F602', 1, 'ORDER',     'O105',   NULL),
    ('F602', 2, 'STATEMENT', 'STMT105',NULL),
    ('F602', 3, 'DATE',      NULL,     '2026-02-05')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-007: DIGITAL_ACCESS_GRANTED + DOWNLOAD_STARTED; klastry: DIGITAL, consent=YES, download=YES
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-007-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-007')
    FROM (VALUES
        ('F701', 'DIGITAL_ACCESS_GRANTED', 2, '2026-02-01 12:00:00'),
        ('F702', 'DOWNLOAD_STARTED',       2, '2026-02-01 12:05:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F701', 0, 'ORDER', 'O106', NULL),
    ('F701', 1, 'DATE',  NULL,   '2026-02-01'),
    ('F702', 0, 'ORDER', 'O106', NULL),
    ('F702', 1, 'DATE',  NULL,   '2026-02-01')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-008: DELIVERED + WITHDRAWAL_STATEMENT_SUBMITTED; brak RETURNED i RETURN_PROOF_PROVIDED
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-008-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-008')
    FROM (VALUES
        ('F801', 'DELIVERED',                      3, '2026-02-01 14:00:00'),
        ('F802', 'WITHDRAWAL_STATEMENT_SUBMITTED', 4, '2026-02-10 10:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F801', 0, 'ORDER',     'O107',   NULL),
    ('F801', 1, 'DELIVERY',  'DEL107', NULL),
    ('F801', 2, 'DATE',      NULL,     '2026-02-01'),
    ('F802', 0, 'CUSTOMER',  'CUST8',  NULL),
    ('F802', 1, 'ORDER',     'O107',   NULL),
    ('F802', 2, 'STATEMENT', 'STMT107',NULL),
    ('F802', 3, 'DATE',      NULL,     '2026-02-10')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-009: COMPLAINT_SUBMITTED(2026-02-01); brak COMPLAINT_RESPONSE_SENT → przekroczone 14 dni
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-009-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-009')
    FROM (VALUES
        ('F901', 'COMPLAINT_SUBMITTED', 4, '2026-02-01 09:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F901', 0, 'CUSTOMER',  'CUST9',   NULL),
    ('F901', 1, 'ORDER',     'O108',    NULL),
    ('F901', 2, 'COMPLAINT', 'COMP900', NULL),
    ('F901', 3, 'DATE',      NULL,      '2026-02-01')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- TC-010: COUPON_APPLIED × 2 (SAVE10, EXTRA5)
WITH ins AS (
    INSERT INTO facts (fact_id, predicate, arity, status, truth_value, truth_confidence, event_time, source_id, case_id)
    SELECT f.fid, f.pred, f.arity, 'observed', 'T', 1.0, f.ts::TIMESTAMPTZ, 'TC-010-TEXT',
           (SELECT id FROM cases WHERE case_id = 'TC-010')
    FROM (VALUES
        ('F1001', 'COUPON_APPLIED', 4, '2026-03-01 10:00:00'),
        ('F1002', 'COUPON_APPLIED', 4, '2026-03-01 10:00:00')
    ) AS f(fid, pred, arity, ts)
    RETURNING id, fact_id
)
INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.pos, a.role, a.eid, a.lv FROM ins f
JOIN (VALUES
    ('F1001', 0, 'CUSTOMER', 'CUST10',     NULL),
    ('F1001', 1, 'ORDER',    'O109',       NULL),
    ('F1001', 2, 'COUPON',   'CPNSAVE10',  NULL),
    ('F1001', 3, 'DATE',     NULL,         '2026-03-01'),
    ('F1002', 0, 'CUSTOMER', 'CUST10',     NULL),
    ('F1002', 1, 'ORDER',    'O109',       NULL),
    ('F1002', 2, 'COUPON',   'CPNEXTRA5',  NULL),
    ('F1002', 3, 'DATE',     NULL,         '2026-03-01')
) AS a(fid, pos, role, eid, lv) ON f.fact_id = a.fid;

-- ============================================================
-- CLUSTER STATES (manual clamps)
-- Dziedziny:
--   customer_type:         [CONSUMER=0, BUSINESS=1]
--   product_type:          [PHYSICAL=0, DIGITAL=1, CUSTOM=2]
--   digital_consent:       [YES=0, NO=1, UNKNOWN=2]
--   download_started_flag: [YES=0, NO=1, UNKNOWN=2]
--   coupon_stackable:      [YES=0, NO=1]
-- ============================================================

DO $$
DECLARE
    c_customer_type         INTEGER := (SELECT id FROM cluster_definitions WHERE name='customer_type');
    c_product_type          INTEGER := (SELECT id FROM cluster_definitions WHERE name='product_type');
    c_digital_consent       INTEGER := (SELECT id FROM cluster_definitions WHERE name='digital_consent');
    c_download_started_flag INTEGER := (SELECT id FROM cluster_definitions WHERE name='download_started_flag');
    c_coupon_stackable      INTEGER := (SELECT id FROM cluster_definitions WHERE name='coupon_stackable');

    e_cust4     INTEGER := (SELECT id FROM entities WHERE entity_id='CUST4');
    e_cust5     INTEGER := (SELECT id FROM entities WHERE entity_id='CUST5');
    e_cust6     INTEGER := (SELECT id FROM entities WHERE entity_id='CUST6');
    e_cust7     INTEGER := (SELECT id FROM entities WHERE entity_id='CUST7');
    e_cust8     INTEGER := (SELECT id FROM entities WHERE entity_id='CUST8');
    e_prod105   INTEGER := (SELECT id FROM entities WHERE entity_id='PROD105');
    e_prod106   INTEGER := (SELECT id FROM entities WHERE entity_id='PROD106');
    e_o106      INTEGER := (SELECT id FROM entities WHERE entity_id='O106');
    e_cpnsave10 INTEGER := (SELECT id FROM entities WHERE entity_id='CPNSAVE10');

    tc004 INTEGER := (SELECT id FROM cases WHERE case_id='TC-004');
    tc005 INTEGER := (SELECT id FROM cases WHERE case_id='TC-005');
    tc006 INTEGER := (SELECT id FROM cases WHERE case_id='TC-006');
    tc007 INTEGER := (SELECT id FROM cases WHERE case_id='TC-007');
    tc008 INTEGER := (SELECT id FROM cases WHERE case_id='TC-008');
    tc010 INTEGER := (SELECT id FROM cases WHERE case_id='TC-010');
BEGIN

    -- TC-004: CUST4 = CONSUMER
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_cust4, c_customer_type, tc004, ARRAY[10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-005: CUST5 = CONSUMER
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_cust5, c_customer_type, tc005, ARRAY[10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-006: CUST6 = CONSUMER
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_cust6, c_customer_type, tc006, ARRAY[10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-006: PROD105 = CUSTOM → [PHYSICAL=-10, DIGITAL=-10, CUSTOM=+10]
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_prod105, c_product_type, tc006, ARRAY[-10.0, -10.0, 10.0], TRUE, TRUE, 'manual');

    -- TC-007: CUST7 = CONSUMER
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_cust7, c_customer_type, tc007, ARRAY[10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-007: PROD106 = DIGITAL → [PHYSICAL=-10, DIGITAL=+10, CUSTOM=-10]
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_prod106, c_product_type, tc007, ARRAY[-10.0, 10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-007: digital_consent(O106) = YES → [YES=+10, NO=-10, UNKNOWN=-10]
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_o106, c_digital_consent, tc007, ARRAY[10.0, -10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-007: download_started_flag(O106) = YES
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_o106, c_download_started_flag, tc007, ARRAY[10.0, -10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-008: CUST8 = CONSUMER
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_cust8, c_customer_type, tc008, ARRAY[10.0, -10.0], TRUE, TRUE, 'manual');

    -- TC-010: coupon_stackable(CPNSAVE10) = NO → [YES=-10, NO=+10]
    INSERT INTO cluster_states (entity_id, cluster_id, case_id, logits, is_clamped, clamp_hard, clamp_source)
    VALUES (e_cpnsave10, c_coupon_stackable, tc010, ARRAY[-10.0, 10.0], TRUE, TRUE, 'manual');

END $$;

-- ============================================================
-- CASE QUERIES
-- ============================================================

INSERT INTO case_queries (case_id, query, expected_result, notes)
SELECT c.id, q.query, q.expected_result, q.notes
FROM cases c
JOIN (VALUES
    ('TC-002', 'may_cancel_for_nonpayment(O101)',          'proved',     'Przelew (prepaid) + brak płatności w 48h'),
    ('TC-003', 'may_cancel_for_nonpayment(O102)',          'not_proved', 'COD nie jest metodą prepaid'),
    ('TC-004', 'can_withdraw(CUST4, O103)',                'proved',     'Konsument, oświadczenie 10 dni po dostawie – w terminie'),
    ('TC-005', 'can_withdraw(CUST5, O104)',                'not_proved', 'Oświadczenie 19 dni po dostawie – po terminie 14-dniowym'),
    ('TC-006', 'ab_withdraw(CUST6, O105)',                 'proved',     'Produkt na zamówienie → wyjątek od prawa odstąpienia'),
    ('TC-006', 'can_withdraw(CUST6, O105)',                'not_proved', 'ab_withdraw blokuje can_withdraw (NAF niespełniony)'),
    ('TC-007', 'ab_withdraw(CUST7, O106)',                 'proved',     'Cyfrowy + zgoda + rozpoczęte pobieranie → wyjątek'),
    ('TC-007', 'can_withdraw(CUST7, O106)',                'not_proved', 'ab_withdraw blokuje can_withdraw'),
    ('TC-008', 'ab_refund_hold(O107)',                     'proved',     'Brak zwrotu i brak dowodu nadania → wstrzymanie zwrotu'),
    ('TC-008', 'refund_due(O107)',                         'blocked',    'ab_refund_hold aktywny → refund_due nie wyprowadzalne'),
    ('TC-009', 'complaint_accepted(C900)',                 'proved',     'Brak odpowiedzi sklepu w 14 dniach → reklamacja uznana'),
    ('TC-010', 'cannot_stack(CPNSAVE10, CPNEXTRA5)',       'proved',     'SAVE10 ma coupon_stackable=NO'),
    ('TC-010', 'coupon_valid_for_order(CPNEXTRA5, O109)',  'not_proved', 'EXTRA5 wymaga min. 200 zł, koszyk O109 to 150 zł')
) AS q(case_id, query, expected_result, notes) ON c.case_id = q.case_id;
