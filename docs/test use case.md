## 1) Testowy fragment „Regulaminu sklepu internetowego” (szerszy zakres)

**§1. Definicje i role**

1. Sklep – podmiot prowadzący sprzedaż w Serwisie.
2. Klient – osoba składająca Zamówienie. Konsument – Klient będący osobą fizyczną dokonującą zakupu niezwiązanego bezpośrednio z działalnością gospodarczą. Przedsiębiorca – Klient dokonujący zakupu w związku z działalnością gospodarczą.

**§2. Składanie zamówień i płatność**

1. Umowa sprzedaży zostaje zawarta z chwilą potwierdzenia przyjęcia Zamówienia przez Sklep (e-mail).
2. Dostępne metody płatności: karta, przelew, BLIK, płatność przy odbiorze.
3. Brak płatności w terminie 48 godzin od złożenia Zamówienia (dla metod przedpłaconych) może skutkować anulowaniem Zamówienia.

**§3. Dostawa**

1. Dostawa realizowana jest na adres wskazany przez Klienta.
2. Termin dostawy wynosi zwykle 1–3 dni robocze, o ile opis produktu nie stanowi inaczej.
3. Ryzyko przypadkowej utraty lub uszkodzenia towaru przechodzi na Konsumenta z chwilą doręczenia.

**§4. Produkty cyfrowe**

1. Dostarczenie treści cyfrowych następuje poprzez udostępnienie linku do pobrania lub aktywację dostępu.
2. Klient może utracić prawo odstąpienia od umowy, jeżeli rozpoczął pobieranie lub korzystanie z treści cyfrowej za wyraźną zgodą.

**§5. Odstąpienie od umowy i zwroty**

1. Konsument może odstąpić od umowy w terminie 14 dni od doręczenia.
2. Prawo odstąpienia nie przysługuje dla: produktów wykonanych na zamówienie; treści cyfrowych po rozpoczęciu pobierania, jeśli Klient wyraził zgodę.
3. Zwrot środków następuje do 14 dni od otrzymania oświadczenia, jednak Sklep może wstrzymać zwrot do czasu otrzymania towaru lub dowodu odesłania.
4. Koszt odesłania ponosi Klient, chyba że towar jest wadliwy lub Sklep zgodził się pokryć koszt.

**§6. Reklamacje (rękojmia/gwarancja) i odpowiedź**

1. Reklamacje rozpatrywane są w terminie 14 dni od zgłoszenia. Brak odpowiedzi oznacza uznanie reklamacji.
2. Sklep może żądać dostarczenia dowodu zakupu i opisu wady.

**§7. Program lojalnościowy i kupony**

1. Kupony rabatowe mają datę ważności i mogą mieć minimalną wartość koszyka.
2. Kupony nie łączą się, chyba że regulamin kuponu stanowi inaczej.
3. Punkty lojalnościowe wygasają po 12 miesiącach nieaktywności.

**§8. Konto i bezpieczeństwo**

1. Klient ponosi odpowiedzialność za udostępnienie hasła.
2. Sklep może zablokować konto w razie podejrzenia nadużyć (np. chargeback fraud), do czasu wyjaśnienia.

**§9. Obciążenia zwrotne (chargeback) i spory płatnicze**

1. W przypadku chargebacku Sklep może wstrzymać realizację kolejnych Zamówień Klienta do czasu wyjaśnienia sporu.
2. W razie uznania chargebacku środki uznaje się za zwrócone Klientowi, a Zamówienie może zostać oznaczone jako sporne.

---

## 2) Ontologia: predykaty i role (minimalna, ale pokrywająca cały regulamin)

### 2.1. Typy encji

* `CUSTOMER`, `ORDER`, `PRODUCT`, `PAYMENT`, `DELIVERY`, `STATEMENT`, `RETURN_SHIPMENT`, `COMPLAINT`, `COUPON`, `ACCOUNT`, `CHARGEBACK`, `DATE`

### 2.2. Predykaty n-arne (reifikowane fakty/zdarzenia)

Poniżej lista predykatów i ról argumentów (arity > 1):

1. `ORDER_PLACED`

* roles: `CUSTOMER`, `ORDER`, `DATE`

2. `ORDER_ACCEPTED` (potwierdzenie przyjęcia)

* roles: `STORE`, `ORDER`, `DATE`

3. `PAYMENT_SELECTED`

* roles: `ORDER`, `PAYMENT_METHOD`

4. `PAYMENT_MADE`

* roles: `ORDER`, `PAYMENT`, `DATE`, `AMOUNT`

5. `PAYMENT_DUE_BY`

* roles: `ORDER`, `DATE`

6. `ORDER_CANCELLED`

* roles: `STORE`, `ORDER`, `DATE`, `REASON`

7. `DELIVERED`

* roles: `ORDER`, `DELIVERY`, `DATE`

8. `DIGITAL_ACCESS_GRANTED`

* roles: `ORDER`, `DATE`

9. `DOWNLOAD_STARTED`

* roles: `ORDER`, `DATE`

10. `WITHDRAWAL_STATEMENT_SUBMITTED`

* roles: `CUSTOMER`, `ORDER`, `STATEMENT`, `DATE`

11. `RETURNED`

* roles: `ORDER`, `RETURN_SHIPMENT`, `DATE`

12. `RETURN_PROOF_PROVIDED`

* roles: `ORDER`, `DATE`

13. `REFUND_ISSUED`

* roles: `ORDER`, `PAYMENT`, `DATE`, `AMOUNT`

14. `COMPLAINT_SUBMITTED`

* roles: `CUSTOMER`, `ORDER`, `COMPLAINT`, `DATE`

15. `COMPLAINT_RESPONSE_SENT`

* roles: `STORE`, `COMPLAINT`, `DATE`

16. `COUPON_APPLIED`

* roles: `CUSTOMER`, `ORDER`, `COUPON`, `DATE`

17. `ACCOUNT_BLOCKED`

* roles: `STORE`, `ACCOUNT`, `DATE`, `REASON`

18. `CHARGEBACK_OPENED`

* roles: `CUSTOMER`, `ORDER`, `CHARGEBACK`, `DATE`

19. `CHARGEBACK_RESOLVED`

* roles: `CHARGEBACK`, `DATE`, `RESULT`  (np. `WON_BY_CUSTOMER`, `WON_BY_STORE`)

### 2.3. Predykaty unarne/klastry (softmax)

Dla encji (zwykle `CUSTOMER`, `ORDER`, `PRODUCT`, `COUPON`, `ACCOUNT`):

* `customer_type(customer) ∈ {CONSUMER, BUSINESS}`
* `payment_method(order) ∈ {CARD, TRANSFER, BLIK, COD}`
* `order_status(order) ∈ {PLACED, ACCEPTED, PAID, CANCELLED, DELIVERED, DISPUTED}`
* `product_type(product) ∈ {PHYSICAL, DIGITAL, CUSTOM}`
* `defective(order) ∈ {YES, NO, UNKNOWN}`
* `store_pays_return(order) ∈ {YES, NO, UNKNOWN}`
* `digital_consent(order) ∈ {YES, NO, UNKNOWN}`
* `download_started_flag(order) ∈ {YES, NO, UNKNOWN}`  (jeśli nie chcesz daty)
* `coupon_stackable(coupon) ∈ {YES, NO}`
* `account_status(account) ∈ {ACTIVE, BLOCKED}`
* `chargeback_status(order) ∈ {NONE, OPEN, RESOLVED_CUSTOMER, RESOLVED_STORE}`

---

## 3) Reguły (przykładowy rdzeń Horn + NAF + wyjątki)

Poniżej tylko kluczowe reguły, które pokrywają przypadki testowe:

### 3.1. Zawarcie umowy

* `contract_formed(o) :- order_accepted(_,o,da).`

### 3.2. Anulowanie za brak płatności (48h)

* `may_cancel_for_nonpayment(o) :- payment_method(o,m), prepaid(m), order_placed(c,o,d0), not paid_within_48h(o,d0).`
* `prepaid(m) :- m=CARD.`
* `prepaid(m) :- m=TRANSFER.`
* `prepaid(m) :- m=BLIK.`
  (COD nie jest prepaid.)

### 3.3. Odstąpienie (default + wyjątki)

* `can_withdraw(c,o) :- customer_type(c)=CONSUMER, delivered(o,d,dd), withdrawal_statement_submitted(c,o,s,ds), within_14_days(ds,dd), not ab_withdraw(c,o).`
* `ab_withdraw(c,o) :- product_type(p)=CUSTOM, order_contains(o,p).`
* `ab_withdraw(c,o) :- product_type(p)=DIGITAL, order_contains(o,p), digital_consent(o)=YES, download_started_flag(o)=YES.`

### 3.4. Zwrot środków i wstrzymanie

* `refund_due(o) :- withdrawal_statement_submitted(c,o,s,ds), not ab_refund_hold(o).`
* `ab_refund_hold(o) :- not returned_or_proof(o).`
* `returned_or_proof(o) :- returned(o,rs,dr).`
* `returned_or_proof(o) :- return_proof_provided(o,dp).`

### 3.5. Reklamacja uznana przez brak odpowiedzi

* `complaint_accepted(comp) :- complaint_submitted(c,o,comp,dc), not responded_in_14_days(comp,dc).`
* `responded_in_14_days(comp,dc) :- complaint_response_sent(store,comp,dr), within_14_days(dr,dc).`

### 3.6. Koszt odesłania

* `customer_pays_return(o) :- not ab_customer_pays_return(o).`
* `ab_customer_pays_return(o) :- defective(o)=YES.`
* `ab_customer_pays_return(o) :- store_pays_return(o)=YES.`

### 3.7. Kupony

* `coupon_valid_for_order(cpn,o) :- coupon_not_expired(cpn, today), meets_min_basket(cpn,o).`
* `cannot_stack(cpn1,cpn2) :- coupon_stackable(cpn1)=NO.`
* `cannot_stack(cpn1,cpn2) :- coupon_stackable(cpn2)=NO.`

### 3.8. Chargeback → ograniczenia

* `may_hold_future_orders(cust) :- chargeback_opened(cust,o,cb,dt), not chargeback_resolved(cb,dr,res).`

---

## 4) 10 „case textów” (teksty spraw), do testów end-to-end

Każdy case to krótki opis, który ekstraktor powinien zamienić na fakty. Do każdego podaję oczekiwane zapytania/wnioski.

### Case 1: Zawarcie umowy

**Text:**
„Złożyłem zamówienie O100 1 lutego. Dostałem maila od sklepu 1 lutego, że zamówienie zostało przyjęte.”
**Oczekiwane:**

* `contract_formed(O100)` = proved

### Case 2: Brak płatności 48h (prepaid)

**Text:**
„Zamówienie O101 złożyłem 1 marca, wybrałem przelew. Nie opłaciłem go do 4 marca. Sklep anulował zamówienie 4 marca.”
**Oczekiwane:**

* `may_cancel_for_nonpayment(O101)` = proved
* `order_status(O101)=CANCELLED` (jeśli wyciągasz)

### Case 3: COD – brak podstawy do anulowania za brak płatności

**Text:**
„Zamówienie O102 złożyłem 1 marca, płatność przy odbiorze. Sklep anulował je 3 marca, bo nie zapłaciłem.”
**Oczekiwane:**

* `may_cancel_for_nonpayment(O102)` = not proved (lub rejected przez constrainty polityki)

### Case 4: Odstąpienie konsumenta w terminie

**Text:**
„Jestem konsumentem. O103 doręczono 1 lutego. Złożyłem oświadczenie o odstąpieniu 10 lutego.”
**Oczekiwane:**

* `can_withdraw(CUST, O103)` = proved

### Case 5: Odstąpienie po terminie

**Text:**
„Jestem konsumentem. O104 doręczono 1 lutego. Oświadczenie o odstąpieniu wysłałem 20 lutego.”
**Oczekiwane:**

* `can_withdraw(CUST, O104)` = not proved (przekroczone 14 dni)

### Case 6: Wyjątek – produkt na zamówienie

**Text:**
„Jestem konsumentem. O105 to meble na wymiar. Doręczono 1 lutego. 5 lutego odstąpiłem od umowy.”
**Oczekiwane:**

* `ab_withdraw(CUST,O105)` = proved
* `can_withdraw(CUST,O105)` = blocked (NAF niespełnione)

### Case 7: Wyjątek – cyfrowy + zgoda + rozpoczęcie pobierania

**Text:**
„Kupiłem produkt cyfrowy w O106. Zgodziłem się na rozpoczęcie świadczenia i zacząłem pobieranie tego samego dnia. Następnego dnia chcę odstąpić.”
**Oczekiwane:**

* `ab_withdraw(CUST,O106)` = proved
* `can_withdraw(CUST,O106)` = blocked

### Case 8: Refund hold – brak zwrotu i brak dowodu

**Text:**
„O107 doręczono 1 lutego, 10 lutego odstąpiłem. Sklep nie zwraca pieniędzy, bo jeszcze nie odesłałem towaru i nie mam potwierdzenia nadania.”
**Oczekiwane:**

* `refund_due(O107)` = blocked przez `ab_refund_hold(O107)` proved

### Case 9: Reklamacja uznana przez brak odpowiedzi

**Text:**
„Zgłosiłem reklamację C900 do zamówienia O108 1 lutego. Do 20 lutego sklep nie odpowiedział.”
**Oczekiwane:**

* `complaint_accepted(C900)` = proved (przekroczone 14 dni)

### Case 10: Kupony nie łączą się + minimalny koszyk

**Text:**
„W O109 próbowałem użyć dwóch kuponów: SAVE10 i EXTRA5. Regulamin SAVE10 mówi, że nie łączy się z innymi. EXTRA5 wymaga minimum 200 zł, a mój koszyk to 150 zł.”
**Oczekiwane:**

* `coupon_valid_for_order(EXTRA5,O109)` = not proved
* `cannot_stack(SAVE10,EXTRA5)` = proved
* (opcjonalnie) wniosek: tylko jeden kupon może być zastosowany, a EXTRA5 i tak nieważny

---

## 5) Jak to testować w Twoim systemie (krótko, deterministycznie)

1. **Extractor**: tekst sprawy → encje (order/customer/coupon) + fakty reifikowane + clampy klastrów.
2. **Neural proposer** (opcjonalnie na start): uzupełnia brakujące klastry (np. `customer_type`) z pamięci.
3. **Symbolic verifier**: kompiluje fakty/reguły i dowodzi zapytań:

   * `can_withdraw`, `refund_due`, `complaint_accepted`, `may_cancel_for_nonpayment`, `cannot_stack`.
4. **Provenance**: dla każdego proved zwraca proof DAG: fakty wejściowe + reguły + sprawdzenia NAF.

## Dodatkowe case’y (§8 konto i bezpieczeństwo, §9 chargeback)

Poniżej 8 dopisanych case textów (11–18), żeby przetestować: blokadę konta, chargeback, wstrzymanie kolejnych zamówień, rozstrzygnięcie sporu oraz deterministyczne provenance.

---

### Case 11: Podejrzenie nadużyć → blokada konta

**Text:**
„Sklep zablokował moje konto A1 5 marca z powodu podejrzenia nadużyć (podejrzenie fraud).”
**Oczekiwane:**

* `account_status(A1)=BLOCKED` (clamp)
* `ACCOUNT_BLOCKED(store, A1, 2026-03-05, FRAUD_SUSPECT)` = observed
* (opcjonalnie reguła) `may_hold_orders_for_account(A1)` = proved

---

### Case 12: Konto zablokowane → ograniczenie składania zamówień

**Text:**
„Moje konto A2 jest zablokowane od 1 marca. 3 marca próbowałem złożyć zamówienie O200, ale system odrzucił.”
**Oczekiwane:**

* `account_status(A2)=BLOCKED`
* `ORDER_PLACED(cust, O200, 2026-03-03)` = observed
* wniosek (jeśli dodasz regułę polityki): `order_allowed(A2,O200)` = not proved / blocked

**Minimalna reguła polityki (opcjonalna):**
`order_blocked_by_account(o) :- order_placed(c,o,d), account_of_customer(c,a), account_status(a)=BLOCKED.`

---

### Case 13: Chargeback otwarty → sklep może wstrzymać kolejne zamówienia

**Text:**
„Otworzyłem chargeback CB1 do zamówienia O201 10 marca.”
**Oczekiwane:**

* `CHARGEBACK_OPENED(cust, O201, CB1, 2026-03-10)` = observed
* `may_hold_future_orders(cust)` = proved (default z §9)

---

### Case 14: Chargeback otwarty, potem rozwiązany na korzyść sklepu

**Text:**
„Chargeback CB2 do O202 otworzyłem 1 marca. 20 marca bank uznał reklamację sklepu (chargeback przegrany przez klienta).”
**Oczekiwane:**

* `CHARGEBACK_RESOLVED(CB2, 2026-03-20, WON_BY_STORE)` = observed
* `may_hold_future_orders(cust)` = not proved (bo spór rozwiązany)
* `chargeback_status(O202)=RESOLVED_STORE` (jeśli prowadzisz taki klaster)

---

### Case 15: Chargeback otwarty, rozwiązany na korzyść klienta → środki uznaje się za zwrócone

**Text:**
„Chargeback CB3 do O203 otworzyłem 1 marca. 15 marca bank uznał chargeback na moją korzyść.”
**Oczekiwane:**

* `CHARGEBACK_RESOLVED(CB3, 2026-03-15, WON_BY_CUSTOMER)` = observed
* wniosek polityki: `funds_considered_refunded(O203)` = proved
* (opcjonalnie) `order_status(O203)=DISPUTED` lub `chargeback_status(O203)=RESOLVED_CUSTOMER`

**Minimalna reguła (opcjonalna):**
`funds_considered_refunded(o) :- chargeback_opened(c,o,cb,_), chargeback_resolved(cb,_,WON_BY_CUSTOMER).`

---

### Case 16: Chargeback + próba nowego zamówienia (wstrzymanie)

**Text:**
„Mam otwarty chargeback CB4. 2 dni później złożyłem nowe zamówienie O204 i sklep napisał, że wstrzymuje realizację do wyjaśnienia sporu.”
**Oczekiwane:**

* `may_hold_future_orders(cust)` = proved
* `ORDER_PLACED(cust,O204,...)` = observed
* wniosek polityki: `fulfillment_on_hold(O204)` = proved

**Minimalna reguła (opcjonalna):**
`fulfillment_on_hold(o2) :- order_placed(c,o2,_), may_hold_future_orders(c).`

---

### Case 17: Hasło udostępnione osobie trzeciej (odpowiedzialność klienta)

**Text:**
„Udostępniłem hasło do konta A3 znajomemu. Ktoś złożył zamówienie O205 z mojego konta.”
**Oczekiwane:**

* `password_shared(A3)=YES` (klaster)
* wniosek (polityka): `customer_responsible_for_password(A3)` = proved

**Minimalna reguła (opcjonalna):**
`customer_responsible_for_password(a) :- password_shared(a)=YES.`

---

### Case 18: Sprzeczne źródła o statusie konta (test konfliktów + provenance)

**Text:**
„System mówi, że konto A4 jest aktywne, ale mail ze sklepu z 1 marca informuje, że konto jest zablokowane.”
**Oczekiwane:**

* dwa fakty obserwowane z różnym `source_rank`/timestamp
* wniosek zależny od polityki konfliktów: np. wybór najnowszego lub najbardziej zaufanego źródła
* provenance powinno wskazać, które źródło wygrało i dlaczego (priorytet/timestamp)

---

## Minimalne dopisane predykaty/klastry do ontologii (żeby to pokryć)

### Predykaty n-arne (reifikowane)

* `PASSWORD_SHARED` (jeśli wolisz zdarzenie): roles `ACCOUNT`, `DATE`
  albo klaster `password_shared(account) ∈ {YES,NO,UNKNOWN}`

### Klastry (unary)

* `password_shared(account) ∈ {YES, NO, UNKNOWN}`
* `fulfillment_status(order) ∈ {NORMAL, ON_HOLD}` (opcjonalnie)
* `funds_considered_refunded(order) ∈ {YES, NO, UNKNOWN}` (opcjonalnie jako klaster, zamiast predykatu)

---

## Proponowane zapytania testowe dla nowych case’ów

* `may_hold_future_orders(customer)`
* `fulfillment_on_hold(order)`
* `funds_considered_refunded(order)`
* `customer_responsible_for_password(account)`
* `account_status(account)`


