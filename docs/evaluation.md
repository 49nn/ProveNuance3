# Evaluation Plan: Accuracy, Coverage, Calibration, Cost, Stability

## 1. Cel
Ten dokument definiuje metryki jakości dla ProveNuance3, sposób ich liczenia oraz progi akceptacji przed release.

## 2. Jednostki oceny
1. `case_query`: para `(case_id, query)` z tabeli `case_queries` i oczekiwanym wynikiem.
2. `fact_extraction`: fakt wyekstrahowany z tekstu (`observed`) porównany do gold.
3. `cluster_extraction`: stan klastra wyekstrahowany z tekstu porównany do gold.
4. `nn_candidate`: kandydat `inferred_candidate` przed symboliczną weryfikacją.

## 3. Metryki końcowe (E2E)

### 3.1 Accuracy i F1 na `case_query`
1. `Exact Accuracy`:
$$
\text{Acc} = \frac{\#\text{poprawnych klasyfikacji}}{\#\text{wszystkich zapytań}}
$$
2. `Macro-F1` dla klas: `proved`, `not_proved`, `blocked`, `unknown`.
3. `Proved Precision/Recall/F1`:
$$
P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F1=\frac{2PR}{P+R}
$$

### 3.2 Coverage
1. `Case Coverage`: odsetek case’ów, które przechodzą cały pipeline bez błędu runtime.
2. `Answer Coverage`: odsetek zapytań z wynikiem różnym od `unknown`.
3. `Proof Coverage`: odsetek `proved`, które mają niepusty `proof_dag`.
4. `Extraction Coverage`: odsetek case’ów z co najmniej jednym faktem i jednym cluster state po `ingest`.

## 4. Metryki modelowe (NN)

### 4.1 Kalibracja
1. `ECE` (Expected Calibration Error) liczony na confidence NN przed SV.
$$
\text{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\left|\text{acc}(B_m)-\text{conf}(B_m)\right|
$$
2. `Brier Score` dla klasyfikacji wieloklasowej `T/F/U`:
$$
\text{Brier}=\frac{1}{n}\sum_{i=1}^{n}\sum_{k}(p_{ik}-y_{ik})^2
$$
3. Kalibrację raportować per-predicate i per-cluster, nie tylko globalnie.

### 4.2 Candidate Quality
1. Precision kandydatów NN, które później stają się `proved`.
2. Recall względem gold faktów, które mogły być zaproponowane przez NN.

## 5. Metryki kosztowe
1. `Latency`:
   - `ingest-text` p50/p95.
   - `run-case` p50/p95.
   - `explain` p50/p95.
2. `LLM Cost`:
   - tokeny wejścia/wyjścia i koszt per case dla ekstraktora i explainera.
3. `Training Cost`:
   - czas `learn-rules` na epokę i całkowity.
4. `Storage Cost`:
   - przyrost rekordów i rozmiaru DB per case (`facts`, `fact_neural_trace`, `proof_steps`).

## 6. Metryki stabilności
1. `Deterministic Replay Pass Rate`: uruchom ten sam case `N` razy; wynik i proof muszą być identyczne.
2. `Seed Robustness` dla `learn-rules`:
   - Jaccard dla zbiorów `rule_id` między seedami.
$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$
3. `Regression Stability`:
   - różnica metryk vs baseline na stałym secie testowym.
4. `Drift`:
   - trend tygodniowy KPI, alarm przy istotnym spadku.

## 7. Minimalny zestaw KPI (Release Gate)
1. `E2E Accuracy` >= `0.90`.
2. `Macro-F1` >= `0.85`.
3. `Answer Coverage` >= `0.95`.
4. `Proof Coverage` >= `0.98` dla klasy `proved`.
5. `ECE` <= `0.08`.
6. `run-case p95` <= `2.0s` na środowisku referencyjnym.
7. `Deterministic Replay Pass Rate` = `1.00`.

Uwaga: progi należy skalibrować po pierwszym pełnym benchmarku na docelowym zbiorze.

## 8. Źródła danych metryk
1. `cases`, `case_queries`.
2. `facts`, `fact_args`, `fact_neural_trace`.
3. `cluster_states`.
4. `proof_runs`, `proof_steps`.
5. Logi CLI (`ingest-*`, `run-case`, `learn-rules`, `explain`).

## 9. Przykładowe zapytania SQL

```sql
-- 1) Proof coverage dla proved factów
SELECT
  COUNT(*) FILTER (WHERE status = 'proved') AS proved_total,
  COUNT(*) FILTER (WHERE status = 'proved' AND proof_id IS NOT NULL) AS proved_with_proof
FROM facts;
```

```sql
-- 2) Udział unknown (answer coverage = 1 - unknown_ratio)
-- Zakłada zapis wyników zapytań do tabeli ewaluacyjnej.
SELECT
  COUNT(*) FILTER (WHERE result = 'unknown')::float / NULLIF(COUNT(*),0) AS unknown_ratio
FROM eval_case_query_results;
```

```sql
-- 3) Koszt storage trace
SELECT COUNT(*) AS trace_rows
FROM fact_neural_trace;
```

## 10. Harmonogram pomiarów
1. Po każdym merge do `main`: smoke metrics na stałym mini-secie.
2. Dziennie: pełny E2E i stabilność.
3. Tygodniowo: raport trendów, kosztów i dryfu.

## 11. Szablon raportu tygodniowego
1. Zakres danych i commit baseline.
2. KPI końcowe: Accuracy, Macro-F1, Coverage, ECE, p95.
3. Największe regresje i ich root cause.
4. Plan działań korygujących na kolejny tydzień.
