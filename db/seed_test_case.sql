-- ProveNuance3 – dane testowe: przypadek TC-001
-- Scenariusz: klient (konsument) składa zamówienie na produkt fizyczny,
--             płaci kartą, otrzymuje dostawę, odstępuje od umowy w ciągu 14 dni,
--             odsyła towar → refund_due(O100) powinno być wyprowadzone przez silnik.

-- ============================================================
-- ŹRÓDŁO (tekst przypadku)
-- ============================================================

INSERT INTO sources (source_id, title, source_type, source_rank) VALUES
    ('TC-001-TEXT', 'Testowy przypadek: zwrot towaru fizycznego', 'case_text', 10);

-- ============================================================
-- CASE
-- ============================================================

INSERT INTO cases (case_id, source_id, title)
SELECT 'TC-001', id, 'Odstąpienie od umowy – produkt fizyczny, karta, zwrot w 14 dniach'
FROM sources WHERE source_id = 'TC-001-TEXT';

-- ============================================================
-- ENCJE
-- ============================================================

INSERT INTO entities (entity_id, entity_type_id, canonical_name) VALUES
    ('CUST1',  (SELECT id FROM entity_types WHERE name='CUSTOMER'),        'Jan Kowalski'),
    ('STORE1', (SELECT id FROM entity_types WHERE name='STORE'),           'Sklep Testowy Sp. z o.o.'),
    ('O100',   (SELECT id FROM entity_types WHERE name='ORDER'),           'Zamówienie #100'),
    ('PROD1',  (SELECT id FROM entity_types WHERE name='PRODUCT'),         'Słuchawki bezprzewodowe XZ-500'),
    ('PAY1',   (SELECT id FROM entity_types WHERE name='PAYMENT'),         'Płatność kartą za O100'),
    ('DEL1',   (SELECT id FROM entity_types WHERE name='DELIVERY'),        'Dostawa DHL #DEL1'),
    ('STMT1',  (SELECT id FROM entity_types WHERE name='STATEMENT'),       'Oświadczenie o odstąpieniu od umowy'),
    ('RS1',    (SELECT id FROM entity_types WHERE name='RETURN_SHIPMENT'), 'Przesyłka zwrotna DHL #RS1');

-- ============================================================
-- FAKTY OBSERWOWANE (ze źródła TC-001-TEXT)
-- ============================================================

-- Używamy CTE żeby łatwo referować id case'u i faktów po wstawieniu

WITH inserted AS (
    INSERT INTO facts
        (fact_id, predicate, arity, status, truth_value, truth_confidence,
         event_time, source_id, case_id)
    SELECT
        f.fact_id, f.predicate, f.arity, 'observed', 'T', 1.0,
        f.event_time::TIMESTAMPTZ, 'TC-001-TEXT',
        (SELECT id FROM cases WHERE case_id = 'TC-001')
    FROM (VALUES
        -- 1. Złożenie zamówienia
        ('F001', 'ORDER_PLACED',                   3, '2026-01-10 09:00:00'),
        -- 2. Przyjęcie zamówienia przez sklep
        ('F002', 'ORDER_ACCEPTED',                 3, '2026-01-10 09:05:00'),
        -- 3. Wybór metody płatności: karta
        ('F003', 'PAYMENT_SELECTED',               2, '2026-01-10 09:00:00'),
        -- 4. Dokonanie płatności
        ('F004', 'PAYMENT_MADE',                   4, '2026-01-10 09:01:00'),
        -- 5. Dostarczenie towaru
        ('F005', 'DELIVERED',                      3, '2026-01-15 14:30:00'),
        -- 6. Oświadczenie o odstąpieniu (5 dni po dostawie, w terminie 14-dniowym)
        ('F006', 'WITHDRAWAL_STATEMENT_SUBMITTED', 4, '2026-01-20 11:00:00'),
        -- 7. Odesłanie towaru
        ('F007', 'RETURNED',                       3, '2026-01-22 10:00:00')
    ) AS f(fact_id, predicate, arity, event_time)
    RETURNING id, fact_id
)

-- ============================================================
-- ARGUMENTY FAKTÓW
-- ============================================================

INSERT INTO fact_args (fact_id, position, role, entity_id, literal_value)
SELECT f.id, a.position, a.role, a.entity_id, a.literal_value
FROM inserted f
JOIN (VALUES
    -- F001: ORDER_PLACED(CUST1, O100, 2026-01-10)
    ('F001', 0, 'CUSTOMER', 'CUST1',  NULL),
    ('F001', 1, 'ORDER',    'O100',   NULL),
    ('F001', 2, 'DATE',     NULL,     '2026-01-10'),

    -- F002: ORDER_ACCEPTED(STORE1, O100, 2026-01-10)
    ('F002', 0, 'STORE',    'STORE1', NULL),
    ('F002', 1, 'ORDER',    'O100',   NULL),
    ('F002', 2, 'DATE',     NULL,     '2026-01-10'),

    -- F003: PAYMENT_SELECTED(O100, card)
    ('F003', 0, 'ORDER',          'O100', NULL),
    ('F003', 1, 'PAYMENT_METHOD', NULL,   'card'),

    -- F004: PAYMENT_MADE(O100, PAY1, 2026-01-10, 299.99)
    ('F004', 0, 'ORDER',   'O100', NULL),
    ('F004', 1, 'PAYMENT', 'PAY1', NULL),
    ('F004', 2, 'DATE',    NULL,   '2026-01-10'),
    ('F004', 3, 'AMOUNT',  NULL,   '299.99'),

    -- F005: DELIVERED(O100, DEL1, 2026-01-15)
    ('F005', 0, 'ORDER',    'O100', NULL),
    ('F005', 1, 'DELIVERY', 'DEL1', NULL),
    ('F005', 2, 'DATE',     NULL,   '2026-01-15'),

    -- F006: WITHDRAWAL_STATEMENT_SUBMITTED(CUST1, O100, STMT1, 2026-01-20)
    ('F006', 0, 'CUSTOMER',  'CUST1',  NULL),
    ('F006', 1, 'ORDER',     'O100',   NULL),
    ('F006', 2, 'STATEMENT', 'STMT1',  NULL),
    ('F006', 3, 'DATE',      NULL,     '2026-01-20'),

    -- F007: RETURNED(O100, RS1, 2026-01-22)
    ('F007', 0, 'ORDER',           'O100', NULL),
    ('F007', 1, 'RETURN_SHIPMENT', 'RS1',  NULL),
    ('F007', 2, 'DATE',            NULL,   '2026-01-22')

) AS a(fact_id, position, role, entity_id, literal_value) ON f.fact_id = a.fact_id;

-- ============================================================
-- QUERY TESTOWE
-- ============================================================

INSERT INTO case_queries (case_id, query, expected_result, notes)
SELECT id, q.query, q.expected_result, q.notes
FROM cases, (VALUES
    ('refund_due(O100)',    'proved',     'Oświadczenie złożone + towar odesłany → zwrot należny'),
    ('can_withdraw(CUST1, O100)', 'proved', 'Konsument, dostawa, oświadczenie w 14 dniach, brak wyjątku'),
    ('contract_formed(O100)',     'proved', 'Zamówienie przyjęte przez sklep')
) AS q(query, expected_result, notes)
WHERE case_id = 'TC-001';
