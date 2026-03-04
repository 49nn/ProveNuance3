-- ProveNuance3 – seed danych ontologii i reguł
-- Zgodny z: schemas/entity.json, schemas/fact.json, schemas/rule.json, schemas/common.json
-- Źródło domeny: docs/test use case.md

-- ============================================================
-- TYPY ENCJI
-- ============================================================

INSERT INTO entity_types (name, description) VALUES
    ('CUSTOMER',        'Klient składający zamówienie'),
    ('ORDER',           'Zamówienie'),
    ('PRODUCT',         'Produkt (fizyczny, cyfrowy lub na zamówienie)'),
    ('PAYMENT',         'Płatność'),
    ('DELIVERY',        'Dostawa'),
    ('STATEMENT',       'Oświadczenie o odstąpieniu od umowy'),
    ('RETURN_SHIPMENT', 'Przesyłka zwrotna'),
    ('COMPLAINT',       'Reklamacja'),
    ('COUPON',          'Kupon rabatowy'),
    ('ACCOUNT',         'Konto klienta'),
    ('CHARGEBACK',      'Obciążenie zwrotne (chargeback)'),
    ('DATE',            'Data / znacznik czasu (literał)'),
    ('AMOUNT',          'Kwota pieniężna (literał)'),
    ('LITERAL',         'Ogólna wartość literalna (enum, string)'),
    ('STORE',           'Sklep – podmiot prowadzący sprzedaż');

-- ============================================================
-- PREDYKATY N-ARNE
-- ============================================================

INSERT INTO predicate_definitions (name, description) VALUES
    ('ORDER_PLACED',                    'Złożenie zamówienia'),
    ('ORDER_ACCEPTED',                  'Potwierdzenie przyjęcia zamówienia przez sklep (e-mail)'),
    ('PAYMENT_SELECTED',                'Wybór metody płatności dla zamówienia'),
    ('PAYMENT_MADE',                    'Dokonanie płatności'),
    ('PAYMENT_DUE_BY',                  'Termin płatności (dla metod przedpłaconych)'),
    ('ORDER_CANCELLED',                 'Anulowanie zamówienia'),
    ('DELIVERED',                       'Doręczenie zamówienia Klientowi'),
    ('DIGITAL_ACCESS_GRANTED',          'Udostępnienie linku / aktywacja treści cyfrowej'),
    ('DOWNLOAD_STARTED',                'Rozpoczęcie pobierania treści cyfrowej'),
    ('WITHDRAWAL_STATEMENT_SUBMITTED',  'Złożenie oświadczenia o odstąpieniu od umowy'),
    ('RETURNED',                        'Odesłanie towaru przez Klienta'),
    ('RETURN_PROOF_PROVIDED',           'Dostarczenie dowodu nadania przesyłki zwrotnej'),
    ('REFUND_ISSUED',                   'Zwrot środków Klientowi'),
    ('COMPLAINT_SUBMITTED',             'Zgłoszenie reklamacji'),
    ('COMPLAINT_RESPONSE_SENT',         'Wysłanie odpowiedzi na reklamację przez sklep'),
    ('COUPON_APPLIED',                  'Zastosowanie kuponu rabatowego do zamówienia'),
    ('ACCOUNT_BLOCKED',                 'Zablokowanie konta przez sklep'),
    ('CHARGEBACK_OPENED',               'Otwarcie procedury chargeback'),
    ('CHARGEBACK_RESOLVED',             'Rozstrzygnięcie procedury chargeback');

-- Role predykatów – zdefiniowane w bloku DO $$ żeby unikać hardkodowania id

DO $$
DECLARE
    p_id INTEGER;

    t_customer        INTEGER := (SELECT id FROM entity_types WHERE name='CUSTOMER');
    t_order           INTEGER := (SELECT id FROM entity_types WHERE name='ORDER');
    t_payment         INTEGER := (SELECT id FROM entity_types WHERE name='PAYMENT');
    t_delivery        INTEGER := (SELECT id FROM entity_types WHERE name='DELIVERY');
    t_statement       INTEGER := (SELECT id FROM entity_types WHERE name='STATEMENT');
    t_return_shipment INTEGER := (SELECT id FROM entity_types WHERE name='RETURN_SHIPMENT');
    t_complaint       INTEGER := (SELECT id FROM entity_types WHERE name='COMPLAINT');
    t_coupon          INTEGER := (SELECT id FROM entity_types WHERE name='COUPON');
    t_account         INTEGER := (SELECT id FROM entity_types WHERE name='ACCOUNT');
    t_chargeback      INTEGER := (SELECT id FROM entity_types WHERE name='CHARGEBACK');
    t_store           INTEGER := (SELECT id FROM entity_types WHERE name='STORE');
    t_date            INTEGER := (SELECT id FROM entity_types WHERE name='DATE');
    t_amount          INTEGER := (SELECT id FROM entity_types WHERE name='AMOUNT');

BEGIN

    -- ORDER_PLACED(CUSTOMER, ORDER, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='ORDER_PLACED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'CUSTOMER', t_customer),
        (p_id, 1, 'ORDER',    t_order),
        (p_id, 2, 'DATE',     t_date);

    -- ORDER_ACCEPTED(STORE, ORDER, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='ORDER_ACCEPTED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'STORE', t_store),
        (p_id, 1, 'ORDER', t_order),
        (p_id, 2, 'DATE',  t_date);

    -- PAYMENT_SELECTED(ORDER, PAYMENT_METHOD:LITERAL)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='PAYMENT_SELECTED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER',          t_order),
        (p_id, 1, 'PAYMENT_METHOD', NULL);

    -- PAYMENT_MADE(ORDER, PAYMENT, DATE, AMOUNT)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='PAYMENT_MADE';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER',   t_order),
        (p_id, 1, 'PAYMENT', t_payment),
        (p_id, 2, 'DATE',    t_date),
        (p_id, 3, 'AMOUNT',  t_amount);

    -- PAYMENT_DUE_BY(ORDER, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='PAYMENT_DUE_BY';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER', t_order),
        (p_id, 1, 'DATE',  t_date);

    -- ORDER_CANCELLED(STORE, ORDER, DATE, REASON:LITERAL)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='ORDER_CANCELLED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'STORE',  t_store),
        (p_id, 1, 'ORDER',  t_order),
        (p_id, 2, 'DATE',   t_date),
        (p_id, 3, 'REASON', NULL);

    -- DELIVERED(ORDER, DELIVERY, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='DELIVERED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER',    t_order),
        (p_id, 1, 'DELIVERY', t_delivery),
        (p_id, 2, 'DATE',     t_date);

    -- DIGITAL_ACCESS_GRANTED(ORDER, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='DIGITAL_ACCESS_GRANTED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER', t_order),
        (p_id, 1, 'DATE',  t_date);

    -- DOWNLOAD_STARTED(ORDER, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='DOWNLOAD_STARTED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER', t_order),
        (p_id, 1, 'DATE',  t_date);

    -- WITHDRAWAL_STATEMENT_SUBMITTED(CUSTOMER, ORDER, STATEMENT, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='WITHDRAWAL_STATEMENT_SUBMITTED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'CUSTOMER',  t_customer),
        (p_id, 1, 'ORDER',     t_order),
        (p_id, 2, 'STATEMENT', t_statement),
        (p_id, 3, 'DATE',      t_date);

    -- RETURNED(ORDER, RETURN_SHIPMENT, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='RETURNED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER',           t_order),
        (p_id, 1, 'RETURN_SHIPMENT', t_return_shipment),
        (p_id, 2, 'DATE',            t_date);

    -- RETURN_PROOF_PROVIDED(ORDER, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='RETURN_PROOF_PROVIDED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER', t_order),
        (p_id, 1, 'DATE',  t_date);

    -- REFUND_ISSUED(ORDER, PAYMENT, DATE, AMOUNT)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='REFUND_ISSUED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'ORDER',   t_order),
        (p_id, 1, 'PAYMENT', t_payment),
        (p_id, 2, 'DATE',    t_date),
        (p_id, 3, 'AMOUNT',  t_amount);

    -- COMPLAINT_SUBMITTED(CUSTOMER, ORDER, COMPLAINT, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='COMPLAINT_SUBMITTED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'CUSTOMER',  t_customer),
        (p_id, 1, 'ORDER',     t_order),
        (p_id, 2, 'COMPLAINT', t_complaint),
        (p_id, 3, 'DATE',      t_date);

    -- COMPLAINT_RESPONSE_SENT(STORE, COMPLAINT, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='COMPLAINT_RESPONSE_SENT';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'STORE',     t_store),
        (p_id, 1, 'COMPLAINT', t_complaint),
        (p_id, 2, 'DATE',      t_date);

    -- COUPON_APPLIED(CUSTOMER, ORDER, COUPON, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='COUPON_APPLIED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'CUSTOMER', t_customer),
        (p_id, 1, 'ORDER',    t_order),
        (p_id, 2, 'COUPON',   t_coupon),
        (p_id, 3, 'DATE',     t_date);

    -- ACCOUNT_BLOCKED(STORE, ACCOUNT, DATE, REASON:LITERAL)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='ACCOUNT_BLOCKED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'STORE',   t_store),
        (p_id, 1, 'ACCOUNT', t_account),
        (p_id, 2, 'DATE',    t_date),
        (p_id, 3, 'REASON',  NULL);

    -- CHARGEBACK_OPENED(CUSTOMER, ORDER, CHARGEBACK, DATE)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='CHARGEBACK_OPENED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'CUSTOMER',   t_customer),
        (p_id, 1, 'ORDER',      t_order),
        (p_id, 2, 'CHARGEBACK', t_chargeback),
        (p_id, 3, 'DATE',       t_date);

    -- CHARGEBACK_RESOLVED(CHARGEBACK, DATE, RESULT:LITERAL)
    SELECT id INTO p_id FROM predicate_definitions WHERE name='CHARGEBACK_RESOLVED';
    INSERT INTO predicate_roles VALUES
        (p_id, 0, 'CHARGEBACK', t_chargeback),
        (p_id, 1, 'DATE',       t_date),
        (p_id, 2, 'RESULT',     NULL);

END $$;

-- ============================================================
-- KLASTRY UNARNE
-- ============================================================

INSERT INTO cluster_definitions (name, entity_type_id, description)
VALUES
    ('customer_type',         (SELECT id FROM entity_types WHERE name='CUSTOMER'), 'Typ klienta: konsument vs przedsiębiorca'),
    ('order_status',          (SELECT id FROM entity_types WHERE name='ORDER'),    'Status zamówienia'),
    ('payment_method',        (SELECT id FROM entity_types WHERE name='ORDER'),    'Metoda płatności wybrana dla zamówienia'),
    ('product_type',          (SELECT id FROM entity_types WHERE name='PRODUCT'),  'Typ produktu'),
    ('defective',             (SELECT id FROM entity_types WHERE name='ORDER'),    'Czy towar jest wadliwy'),
    ('store_pays_return',     (SELECT id FROM entity_types WHERE name='ORDER'),    'Czy sklep zgodził się pokryć koszt odesłania'),
    ('digital_consent',       (SELECT id FROM entity_types WHERE name='ORDER'),    'Zgoda na dostęp do treści cyfrowej'),
    ('download_started_flag', (SELECT id FROM entity_types WHERE name='ORDER'),    'Czy pobieranie treści cyfrowej zostało rozpoczęte'),
    ('coupon_stackable',      (SELECT id FROM entity_types WHERE name='COUPON'),   'Czy kupon można łączyć z innymi kuponami'),
    ('account_status',        (SELECT id FROM entity_types WHERE name='ACCOUNT'),  'Status konta klienta'),
    ('chargeback_status',     (SELECT id FROM entity_types WHERE name='ORDER'),    'Status procedury chargeback dla zamówienia'),
    ('password_shared',       (SELECT id FROM entity_types WHERE name='ACCOUNT'),  'Czy klient udostępnił hasło osobie trzeciej');

-- Wartości dziedzin klastrów

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'CONSUMER'),(1,'BUSINESS')) AS v(pos,val)
WHERE name='customer_type';

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'PLACED'),(1,'ACCEPTED'),(2,'PAID'),(3,'CANCELLED'),(4,'DELIVERED'),(5,'DISPUTED')) AS v(pos,val)
WHERE name='order_status';

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'CARD'),(1,'TRANSFER'),(2,'BLIK'),(3,'COD')) AS v(pos,val)
WHERE name='payment_method';

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'PHYSICAL'),(1,'DIGITAL'),(2,'CUSTOM')) AS v(pos,val)
WHERE name='product_type';

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'YES'),(1,'NO'),(2,'UNKNOWN')) AS v(pos,val)
WHERE name IN ('defective','store_pays_return','digital_consent','download_started_flag','password_shared');

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'YES'),(1,'NO')) AS v(pos,val)
WHERE name='coupon_stackable';

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'ACTIVE'),(1,'BLOCKED')) AS v(pos,val)
WHERE name='account_status';

INSERT INTO cluster_domain_values (cluster_id, position, value)
SELECT id, v.pos, v.val FROM cluster_definitions,
    (VALUES (0,'NONE'),(1,'OPEN'),(2,'RESOLVED_CUSTOMER'),(3,'RESOLVED_STORE')) AS v(pos,val)
WHERE name='chargeback_status';

-- ============================================================
-- MODUŁY REGUŁ
-- ============================================================

INSERT INTO rule_modules (name, description) VALUES
    ('contract',    'Zawarcie umowy sprzedaży'),
    ('payment',     'Płatności, terminy, anulowanie za brak płatności'),
    ('withdrawal',  'Odstąpienie od umowy i wyjątki'),
    ('refund',      'Zwrot środków i wstrzymanie'),
    ('complaint',   'Reklamacje i terminy odpowiedzi'),
    ('return_cost', 'Koszt odesłania towaru'),
    ('coupon',      'Kupony rabatowe i łączenie'),
    ('account',     'Konto klienta i blokady'),
    ('chargeback',  'Procedura chargeback i wstrzymanie zamówień');

-- ============================================================
-- REGUŁY HORN + NAF (rule.json)
-- head/body JSONB: common.json → RuleArg {role, term: {var|const}, type_hint?}
-- body literal:    rule.json   → {literal_type: pos|naf, predicate, args}
-- ============================================================

INSERT INTO rules (rule_id, module_id, head, body, clingo_text, stratum, learned) VALUES

-- §3.1 Zawarcie umowy
('contract.contract_formed',
    (SELECT id FROM rule_modules WHERE name='contract'),
    '{"predicate":"contract_formed","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"order_accepted","args":[{"role":"STORE","term":{"var":"_"}},{"role":"ORDER","term":{"var":"O"}},{"role":"DATE","term":{"var":"_"}}]}]',
    'contract_formed(O) :- order_accepted(_,O,_).',
    0, false),

-- §2.3 Prepaid
('payment.prepaid_card',
    (SELECT id FROM rule_modules WHERE name='payment'),
    '{"predicate":"prepaid","args":[{"role":"METHOD","term":{"const":"card"}}]}',
    '[]',
    'prepaid(card).',
    0, false),

('payment.prepaid_transfer',
    (SELECT id FROM rule_modules WHERE name='payment'),
    '{"predicate":"prepaid","args":[{"role":"METHOD","term":{"const":"transfer"}}]}',
    '[]',
    'prepaid(transfer).',
    0, false),

('payment.prepaid_blik',
    (SELECT id FROM rule_modules WHERE name='payment'),
    '{"predicate":"prepaid","args":[{"role":"METHOD","term":{"const":"blik"}}]}',
    '[]',
    'prepaid(blik).',
    0, false),

-- §2.3 Anulowanie za brak płatności (stratum=1: uses NAF over paid_within_48h)
('payment.may_cancel_for_nonpayment',
    (SELECT id FROM rule_modules WHERE name='payment'),
    '{"predicate":"may_cancel_for_nonpayment","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"payment_method","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"METHOD","term":{"var":"M"}}]},{"literal_type":"pos","predicate":"prepaid","args":[{"role":"METHOD","term":{"var":"M"}}]},{"literal_type":"pos","predicate":"order_placed","args":[{"role":"CUSTOMER","term":{"var":"_"}},{"role":"ORDER","term":{"var":"O"}},{"role":"DATE","term":{"var":"D0"}}]},{"literal_type":"naf","predicate":"paid_within_48h","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"DATE","term":{"var":"D0"}}]}]',
    'may_cancel_for_nonpayment(O) :- payment_method(O,M), prepaid(M), order_placed(_,O,D0), not paid_within_48h(O,D0).',
    1, false),

-- §5.1 Odstąpienie od umowy (stratum=2: NAF over ab_withdraw which is at stratum=1)
('withdrawal.can_withdraw',
    (SELECT id FROM rule_modules WHERE name='withdrawal'),
    '{"predicate":"can_withdraw","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"customer_type","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"TYPE","term":{"const":"consumer"}}]},{"literal_type":"pos","predicate":"delivered","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"DELIVERY","term":{"var":"_"}},{"role":"DATE","term":{"var":"DD"}}]},{"literal_type":"pos","predicate":"withdrawal_statement_submitted","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ORDER","term":{"var":"O"}},{"role":"STATEMENT","term":{"var":"_"}},{"role":"DATE","term":{"var":"DS"}}]},{"literal_type":"pos","predicate":"within_14_days","args":[{"role":"FROM","term":{"var":"DS"}},{"role":"TO","term":{"var":"DD"}}]},{"literal_type":"naf","predicate":"ab_withdraw","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ORDER","term":{"var":"O"}}]}]',
    'can_withdraw(C,O) :- customer_type(C,consumer), delivered(O,_,DD), withdrawal_statement_submitted(C,O,_,DS), within_14_days(DS,DD), not ab_withdraw(C,O).',
    2, false),

-- §5.2 Wyjątek: produkt na zamówienie
('withdrawal.ab_withdraw_custom',
    (SELECT id FROM rule_modules WHERE name='withdrawal'),
    '{"predicate":"ab_withdraw","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"product_type","args":[{"role":"PRODUCT","term":{"var":"P"}},{"role":"TYPE","term":{"const":"custom"}}]},{"literal_type":"pos","predicate":"order_contains","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"PRODUCT","term":{"var":"P"}}]}]',
    'ab_withdraw(C,O) :- product_type(P,custom), order_contains(O,P).',
    1, false),

-- §4.2 Wyjątek: treść cyfrowa + zgoda + pobieranie
('withdrawal.ab_withdraw_digital',
    (SELECT id FROM rule_modules WHERE name='withdrawal'),
    '{"predicate":"ab_withdraw","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"product_type","args":[{"role":"PRODUCT","term":{"var":"P"}},{"role":"TYPE","term":{"const":"digital"}}]},{"literal_type":"pos","predicate":"order_contains","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"PRODUCT","term":{"var":"P"}}]},{"literal_type":"pos","predicate":"digital_consent","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"VALUE","term":{"const":"yes"}}]},{"literal_type":"pos","predicate":"download_started_flag","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"VALUE","term":{"const":"yes"}}]}]',
    'ab_withdraw(C,O) :- product_type(P,digital), order_contains(O,P), digital_consent(O,yes), download_started_flag(O,yes).',
    1, false),

-- §5.3 Zwrot środków
('refund.refund_due',
    (SELECT id FROM rule_modules WHERE name='refund'),
    '{"predicate":"refund_due","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"withdrawal_statement_submitted","args":[{"role":"CUSTOMER","term":{"var":"_"}},{"role":"ORDER","term":{"var":"O"}},{"role":"STATEMENT","term":{"var":"_"}},{"role":"DATE","term":{"var":"_"}}]},{"literal_type":"naf","predicate":"ab_refund_hold","args":[{"role":"ORDER","term":{"var":"O"}}]}]',
    'refund_due(O) :- withdrawal_statement_submitted(_,O,_,_), not ab_refund_hold(O).',
    2, false),

('refund.ab_refund_hold',
    (SELECT id FROM rule_modules WHERE name='refund'),
    '{"predicate":"ab_refund_hold","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"naf","predicate":"returned_or_proof","args":[{"role":"ORDER","term":{"var":"O"}}]}]',
    'ab_refund_hold(O) :- not returned_or_proof(O).',
    1, false),

('refund.returned_or_proof_returned',
    (SELECT id FROM rule_modules WHERE name='refund'),
    '{"predicate":"returned_or_proof","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"returned","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"RETURN_SHIPMENT","term":{"var":"_"}},{"role":"DATE","term":{"var":"_"}}]}]',
    'returned_or_proof(O) :- returned(O,_,_).',
    0, false),

('refund.returned_or_proof_proof',
    (SELECT id FROM rule_modules WHERE name='refund'),
    '{"predicate":"returned_or_proof","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"return_proof_provided","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"DATE","term":{"var":"_"}}]}]',
    'returned_or_proof(O) :- return_proof_provided(O,_).',
    0, false),

-- §6.1 Reklamacja uznana przez brak odpowiedzi
('complaint.responded_in_14_days',
    (SELECT id FROM rule_modules WHERE name='complaint'),
    '{"predicate":"responded_in_14_days","args":[{"role":"COMPLAINT","term":{"var":"Comp"}},{"role":"DATE","term":{"var":"DC"}}]}',
    '[{"literal_type":"pos","predicate":"complaint_response_sent","args":[{"role":"STORE","term":{"const":"store"}},{"role":"COMPLAINT","term":{"var":"Comp"}},{"role":"DATE","term":{"var":"DR"}}]},{"literal_type":"pos","predicate":"within_14_days","args":[{"role":"FROM","term":{"var":"DR"}},{"role":"TO","term":{"var":"DC"}}]}]',
    'responded_in_14_days(Comp,DC) :- complaint_response_sent(store,Comp,DR), within_14_days(DR,DC).',
    0, false),

('complaint.complaint_accepted',
    (SELECT id FROM rule_modules WHERE name='complaint'),
    '{"predicate":"complaint_accepted","args":[{"role":"COMPLAINT","term":{"var":"Comp"}}]}',
    '[{"literal_type":"pos","predicate":"complaint_submitted","args":[{"role":"CUSTOMER","term":{"var":"_"}},{"role":"ORDER","term":{"var":"_"}},{"role":"COMPLAINT","term":{"var":"Comp"}},{"role":"DATE","term":{"var":"DC"}}]},{"literal_type":"naf","predicate":"responded_in_14_days","args":[{"role":"COMPLAINT","term":{"var":"Comp"}},{"role":"DATE","term":{"var":"DC"}}]}]',
    'complaint_accepted(Comp) :- complaint_submitted(_,_,Comp,DC), not responded_in_14_days(Comp,DC).',
    2, false),

-- §5.4 Koszt odesłania
('return_cost.customer_pays_return',
    (SELECT id FROM rule_modules WHERE name='return_cost'),
    '{"predicate":"customer_pays_return","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"naf","predicate":"ab_customer_pays_return","args":[{"role":"ORDER","term":{"var":"O"}}]}]',
    'customer_pays_return(O) :- not ab_customer_pays_return(O).',
    1, false),

('return_cost.ab_customer_pays_return_defective',
    (SELECT id FROM rule_modules WHERE name='return_cost'),
    '{"predicate":"ab_customer_pays_return","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"defective","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"VALUE","term":{"const":"yes"}}]}]',
    'ab_customer_pays_return(O) :- defective(O,yes).',
    0, false),

('return_cost.ab_customer_pays_return_store_agreed',
    (SELECT id FROM rule_modules WHERE name='return_cost'),
    '{"predicate":"ab_customer_pays_return","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"store_pays_return","args":[{"role":"ORDER","term":{"var":"O"}},{"role":"VALUE","term":{"const":"yes"}}]}]',
    'ab_customer_pays_return(O) :- store_pays_return(O,yes).',
    0, false),

-- §7.1-2 Kupony
('coupon.coupon_valid_for_order',
    (SELECT id FROM rule_modules WHERE name='coupon'),
    '{"predicate":"coupon_valid_for_order","args":[{"role":"COUPON","term":{"var":"Cpn"}},{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"coupon_not_expired","args":[{"role":"COUPON","term":{"var":"Cpn"}},{"role":"DATE","term":{"const":"today"}}]},{"literal_type":"pos","predicate":"meets_min_basket","args":[{"role":"COUPON","term":{"var":"Cpn"}},{"role":"ORDER","term":{"var":"O"}}]}]',
    'coupon_valid_for_order(Cpn,O) :- coupon_not_expired(Cpn,today), meets_min_basket(Cpn,O).',
    0, false),

('coupon.cannot_stack_1',
    (SELECT id FROM rule_modules WHERE name='coupon'),
    '{"predicate":"cannot_stack","args":[{"role":"COUPON1","term":{"var":"Cpn1"}},{"role":"COUPON2","term":{"var":"Cpn2"}}]}',
    '[{"literal_type":"pos","predicate":"coupon_stackable","args":[{"role":"COUPON","term":{"var":"Cpn1"}},{"role":"VALUE","term":{"const":"no"}}]}]',
    'cannot_stack(Cpn1,Cpn2) :- coupon_stackable(Cpn1,no).',
    0, false),

('coupon.cannot_stack_2',
    (SELECT id FROM rule_modules WHERE name='coupon'),
    '{"predicate":"cannot_stack","args":[{"role":"COUPON1","term":{"var":"Cpn1"}},{"role":"COUPON2","term":{"var":"Cpn2"}}]}',
    '[{"literal_type":"pos","predicate":"coupon_stackable","args":[{"role":"COUPON","term":{"var":"Cpn2"}},{"role":"VALUE","term":{"const":"no"}}]}]',
    'cannot_stack(Cpn1,Cpn2) :- coupon_stackable(Cpn2,no).',
    0, false),

-- §8.1-2 Konto
('account.order_blocked_by_account',
    (SELECT id FROM rule_modules WHERE name='account'),
    '{"predicate":"order_blocked_by_account","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"order_placed","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ORDER","term":{"var":"O"}},{"role":"DATE","term":{"var":"_"}}]},{"literal_type":"pos","predicate":"account_of_customer","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ACCOUNT","term":{"var":"A"}}]},{"literal_type":"pos","predicate":"account_status","args":[{"role":"ACCOUNT","term":{"var":"A"}},{"role":"VALUE","term":{"const":"blocked"}}]}]',
    'order_blocked_by_account(O) :- order_placed(C,O,_), account_of_customer(C,A), account_status(A,blocked).',
    1, false),

('account.customer_responsible_for_password',
    (SELECT id FROM rule_modules WHERE name='account'),
    '{"predicate":"customer_responsible_for_password","args":[{"role":"ACCOUNT","term":{"var":"A"}}]}',
    '[{"literal_type":"pos","predicate":"password_shared","args":[{"role":"ACCOUNT","term":{"var":"A"}},{"role":"VALUE","term":{"const":"yes"}}]}]',
    'customer_responsible_for_password(A) :- password_shared(A,yes).',
    0, false),

-- §9.1-2 Chargeback
('chargeback.may_hold_future_orders',
    (SELECT id FROM rule_modules WHERE name='chargeback'),
    '{"predicate":"may_hold_future_orders","args":[{"role":"CUSTOMER","term":{"var":"Cust"}}]}',
    '[{"literal_type":"pos","predicate":"chargeback_opened","args":[{"role":"CUSTOMER","term":{"var":"Cust"}},{"role":"ORDER","term":{"var":"_"}},{"role":"CHARGEBACK","term":{"var":"CB"}},{"role":"DATE","term":{"var":"_"}}]},{"literal_type":"naf","predicate":"chargeback_resolved","args":[{"role":"CHARGEBACK","term":{"var":"CB"}},{"role":"DATE","term":{"var":"_"}},{"role":"RESULT","term":{"var":"_"}}]}]',
    'may_hold_future_orders(Cust) :- chargeback_opened(Cust,_,CB,_), not chargeback_resolved(CB,_,_).',
    1, false),

('chargeback.fulfillment_on_hold',
    (SELECT id FROM rule_modules WHERE name='chargeback'),
    '{"predicate":"fulfillment_on_hold","args":[{"role":"ORDER","term":{"var":"O2"}}]}',
    '[{"literal_type":"pos","predicate":"order_placed","args":[{"role":"CUSTOMER","term":{"var":"C"}},{"role":"ORDER","term":{"var":"O2"}},{"role":"DATE","term":{"var":"_"}}]},{"literal_type":"pos","predicate":"may_hold_future_orders","args":[{"role":"CUSTOMER","term":{"var":"C"}}]}]',
    'fulfillment_on_hold(O2) :- order_placed(C,O2,_), may_hold_future_orders(C).',
    2, false),

('chargeback.funds_considered_refunded',
    (SELECT id FROM rule_modules WHERE name='chargeback'),
    '{"predicate":"funds_considered_refunded","args":[{"role":"ORDER","term":{"var":"O"}}]}',
    '[{"literal_type":"pos","predicate":"chargeback_opened","args":[{"role":"CUSTOMER","term":{"var":"_"}},{"role":"ORDER","term":{"var":"O"}},{"role":"CHARGEBACK","term":{"var":"CB"}},{"role":"DATE","term":{"var":"_"}}]},{"literal_type":"pos","predicate":"chargeback_resolved","args":[{"role":"CHARGEBACK","term":{"var":"CB"}},{"role":"DATE","term":{"var":"_"}},{"role":"RESULT","term":{"const":"won_by_customer"}}]}]',
    'funds_considered_refunded(O) :- chargeback_opened(_,O,CB,_), chargeback_resolved(CB,_,won_by_customer).',
    1, false);
