## Cel i intuicja architektury

Chcesz system, który:

1. z tekstu buduje częściową bazę faktów,
2. wykonuje wnioskowanie na grafie/hipergrafie z konkurencją i wyjątkami,
3. uczy się uogólnień (reguł) oraz zapamiętuje informacje jednostkowe (encje),
4. dostarcza provenance i działa deterministycznie,
5. pozwala na symboliczne dowodzenie Horn+NAF z wyjątkami.

Najstabilniejsza architektura to podział na trzy warstwy:

* **Neural proposer**: generuje/uzupełnia fakty i ocenia reguły (miękkie wnioskowanie, ranking).
* **Symbolic verifier**: dowodzi (Horn+NAF), filtruje, generuje formalne provenance.
* **Entity memory**: przechowuje dane instancyjne (adresy, nazwy) i zasila wnioskowanie biasami/clampami.

Uzasadnienie: uogólnienia i wnioskowanie dobrze uczą się parametrycznie, ale dane jednostkowe nie powinny „wchodzić w wagi” (bo powodują przeuczenie i konflikty); natomiast poprawność logiczna i provenance najlepiej zapewnić przez walidator.

---

## Reprezentacja wiedzy

### Encje

Encje (e \in \mathcal{E}) mają:

* identyfikator `entity_id`,
* typ (type(e)),
* embedding pamięci (m_e \in \mathbb{R}^d) do linkowania,
* sloty pamięci (wartości symboliczne + wersje + provenance).

### Fakty unarne jako klastry (zmienne dyskretne)

Dla atrybutu (A) o dziedzinie (D_A={1,\dots,K}) i encji (e) masz klaster:

* zmienna losowa (X_{e,A} \in D_A),
* logity (s_{e,A} \in \mathbb{R}^K),
* rozkład:
  [
  p_{e,A}(k)=\mathrm{softmax}(s_{e,A})*k=\frac{\exp(s*{e,A,k})}{\sum_{j=1}^K \exp(s_{e,A,j})}.
  ]
  **Inhibicja w klastrze** wynika z normalizacji: wzrost jednego (s_{k}) zmniejsza masę pozostałych (p(j)).

### Fakty n-arne jako węzły reifikowane (hipergraf)

Każdy fakt (f) o predykacie (P) i arności (n) reprezentujesz jako węzeł zdarzenia:

* (f = P(a_1,\dots,a_n)),
* role (r_i) (ARG0/AGENT, THEME, ...),
* krawędzie rolowe ((f, r_i, e_i)).

Prawdziwość faktu jako klaster:

* (T_f \in {T,F,U}),
* logity (s_f \in \mathbb{R}^3),
* (p_f = \mathrm{softmax}(s_f)).

Uzasadnienie reifikacji: dowolna arność jest sprowadzona do jednolitej struktury grafowej (hetero-graph), na której można robić message passing.

---

## Model grafowy i matematyka wnioskowania

### Heterogeniczny graf/factor graph

Masz dwa typy węzłów:

* encje (e),
* fakty/klastry (v) (węzły zdarzeń oraz węzły atrybutów).

Krawędzie:

* role: (f \to e) oznaczone rolą,
* reguły/relacje: (u \to v) (typowane).

To odpowiada **factor graph**: zmienne (stany klastrów) i czynniki (zgodności), tylko realizowane jako message passing.

### Aktualizacja logitów (message passing)

Niech (v) będzie węzłem-klastrem z dziedziną (D_v). W każdym kroku (t=0..T-1):

1. Oblicz rozkład bieżący:
   [
   p_v^{(t)}=\mathrm{softmax}(s_v^{(t)}).
   ]

2. Dla każdej krawędzi (u \to v) typu (\tau) wygeneruj wkład do logitów v:
   [
   \Delta s_{u\to v}^{(t)} = p_u^{(t)} W^{+}*{\tau} ;-; p_u^{(t)} W^{-}*{\tau},
   ]
   gdzie:

* (W^{+}_{\tau} \in \mathbb{R}^{|D_u|\times |D_v|}),
* (W^{-}_{\tau}\ge 0) (hamowanie).

3. Zsumuj wkłady:
   [
   s_v^{(t+1)} = s_v^{(t)} + \sum_{u\in \mathcal{N}(v)} \Delta s_{u\to v}^{(t)} + b_v.
   ]

**Inhibicja między klastrami** jest tu jawna: odejmowanie składnika (p_u W^{-}) obniża wybrane logity docelowe.

### Wyjątki jako gating (bardziej stabilne niż ujemne wagi)

Dla reguły default (R) generującej wiadomość (m_R) do celu (v):
[
m_R^{(t)} = p_u^{(t)} W_R,
]
a wyjątek daje bramkę (g\in[0,1]):
[
g^{(t)} = \sigma\big(p_{exc}^{(t)} u\big),
\quad m'_R = (1-g^{(t)})\odot m_R^{(t)},
]
[
s_v^{(t+1)} \mathrel{+}= m'_R.
]
Uzasadnienie: wyjątek gasi wpływ reguły zamiast „walczyć” w logitach z wieloma źródłami.

### Clamp (obserwacje z tekstu i pamięci)

Dla obserwacji (X_v = k) ustawiasz logity:

* twardo: (s_{v,k}=+M), (s_{v,j\neq k}=-M) i zamrażasz,
* miękko: (s_{v,k}\mathrel{+}=M).

Dla pamięci encji analogicznie dodajesz bias (b_v^{mem}).

---

## Uczenie (self-supervised) i uzasadnienie

### Maskowanie jako źródło sygnału

Z tekstu masz obserwacje (C) (clampowane wartości). Losujesz podzbiór (M\subset C) do maskowania. Po wnioskowaniu wymagasz rekonstrukcji:

[
\mathcal{L}*{mask} = - \sum*{v\in M} \log p_v^{(T)}(k_v^{true}).
]

Uzasadnienie: tekst dostarcza „pseudo-etykiet” bez ręcznej adnotacji; wagi uczą się takich zależności, które pozwalają odtwarzać brakujące fakty.

### Regularizacja logiczna (soft constraints)

Dla reguły typu implikacja (A\Rightarrow B) w wersji probabilistycznej:
[
\mathcal{L}*{imp} = \mathbb{E},\max(0,\ p(A)-p(B)).
]
Dla niekompatybilności:
[
\mathcal{L}*{incomp}=\mathbb{E}, p(A),p(B).
]
Dla ograniczenia rozrostu:
[
\mathcal{L}*{sparse}= \sum*{v\notin clamp} H(p_v) \quad \text{lub} \quad \sum_{v\notin clamp}|p_v|_1,
]
gdzie (H) to entropia (używana zależnie od tego, czy chcesz decyzje ostre, czy „unknown”).

Łącznie:
[
\mathcal{L}=\mathcal{L}*{mask}+\lambda\mathcal{L}*{imp}+\mu\mathcal{L}*{incomp}+\beta\mathcal{L}*{sparse}.
]

Uzasadnienie: bez constraintów model ma tendencję do rozwiązań trywialnych (np. nadmierna aktywacja); constraints wprowadzają „kształt” logiki domeny.

### Uczenie wag reguł

Parametry (W^+, W^-) i parametry bramek uczysz gradientowo przez unroll (T) kroków (BPTT). Stabilność poprawiają:

* ograniczenie (W^-\ge0) (np. softplus),
* sparsity na wierszach/kolumnach (W) (ułatwia ekstrakcję reguł),
* ograniczenie liczby kroków (T).

---

## Reguły Horn+NAF i mapowanie z części neuronowej

### Walidator symboliczny

Język:

* Horn clauses,
* NAF jako `not p`,
* wyjątki jako `ab_*`,
* stratyfikacja negacji.

Wnioskowanie:

* bottom-up (Datalog) z negacją stratyfikowaną dla deterministyczności,
* provenience jako DAG wyprowadzeń.

### Propose–verify

* Neural proposer: generuje kandydackie fakty i reguły (ranking).
* Verifier: dowodzi/odrzuca/oznacza unknown.

Uzasadnienie: gwarancje (spójność, wyjaśnialność) trudno uzyskać z samej sieci; walidator daje twarde kryterium akceptacji i formalne provenance.

### Ekstrakcja reguł z macierzy (W)

Dla klastrów (A\to B):

* duże dodatnie (W[a,b]) interpretujesz jako regułę:

  * (B=b \leftarrow A=a).
* aby uniknąć szumu: support, kontrprzykłady, beam search dla reguł złożonych oraz sparsity w treningu.

Ujemne wagi mapujesz raczej na:

* constraints `:- A=a, B=b.`, lub
* wyjątki (gating), jeśli istnieje wykrywalny predykat wyjątku.

---

## Pamięć encji (dane „na pamięć”)

### Co trafia do pamięci

Informacje instancyjne, zwykle:

* nazwy własne, aliasy,
* adresy, numery, identyfikatory,
* relacje „z dokumentu” o wysokiej wiarygodności,
* wersje czasowe i źródła.

### Dlaczego oddzielnie od (W)

* Dane jednostkowe nie generalizują; w wagach powodują przeuczenie i konflikty.
* Pamięć pozwala na:

  * aktualizacje przy strumieniu danych,
  * wersjonowanie i politykę źródeł,
  * korekty entity linking bez „psucia” reguł.

### Jak pamięć zasila wnioskowanie

Pamięć produkuje bias/clamp do logitów:
[
s_v^{(0)} \leftarrow s_v^{(0)} + b_v^{mem}.
]
To jest mechanicznie proste i determinizuje „zapamiętane” fakty.

---

## Provenance

### Symboliczne (proof)

Walidator zwraca:

* listę kroków zastosowania reguł + podstawienia,
* użyte fakty wejściowe (z dokumentów/pamięci),
* wynik NAF dla wyjątków.

To jest provenance w sensie formalnego dowodu.

### Neural trace

Gdy nie ma dowodu, zapisujesz:

* top-k wkładów do logitu celu:

  * (\Delta s_{u\to v}), typ krawędzi, krok (t),
  * opcjonalnie wkłady hamujące.

Uzasadnienie: daje diagnostykę i „explainability” nawet dla hipotez nieudowodnionych.

---

## Deterministyczność: warunki techniczne i logiczne

### Techniczne

* deterministyczna ekstrakcja faktów (brak losowości),
* deterministyczne inference: brak dropout, stałe (T) lub deterministyczny stop,
* deterministyczne zasady match/create w entity linking (stałe progi i tie-break).

### Logiczne

* stratyfikacja negacji w walidatorze,
* jawna polityka czasu i źródeł (żeby konflikty rozstrzygały się jednoznacznie).

Uwaga: system z wyjątkami (NAF) jest z natury niemal zawsze **nemonotoniczny**; to zgodne z celami (defaulty i wyjątki).

---

## Dlaczego ta architektura jest spójna

* **Klastry + softmax**: naturalna realizacja „jedna wartość z wielu” oraz lokalnej inhibicji.
* **Macierze (W)**: odpowiadają potencjałom/zgodnościom między zmiennymi; łatwe do uczenia i do ekstrakcji reguł 1-przesłankowych.
* **Hipergraf faktów n-arnych**: jednolita obsługa dowolnej arności i ról.
* **Gating wyjątków**: stabilne numerycznie i mapowalne na `ab_*` w Horn+NAF.
* **Maskowanie**: zapewnia sygnał uczenia z samego tekstu.
* **Walidator**: daje twarde dowody i filtruje halucynacje.
* **Pamięć encji**: rozwiązuje „uczenie na pamięć” bez niszczenia uogólnień.

