# Biblioteki ML/AI — uzasadnienie doboru

Architektura systemu dzieli się na trzy warstwy: **Neural Proposer**, **Symbolic Verifier** i **Entity Memory**. Poniżej zestawiono biblioteki dopasowane do każdej z nich, a następnie biblioteki przekrojowe.

---

## 1. Warstwa neuronowa — Neural Proposer

### PyTorch
**Rola:** główny framework do budowy modelu message-passing i uczenia gradientowego.

**Uzasadnienie:**
- Obsługuje BPTT przez unroll T kroków (iterowane propagowanie wiadomości na grafie).
- Daje pełną kontrolę nad niestandardową pętlą update `s_v^(t+1) = s_v^(t) + Σ Δs + b_v`.
- Łatwe wymuszenie ograniczeń `W⁻ ≥ 0` przez `softplus` lub `clamp`.
- Autograd obsługuje niestandard. operacje (gating sigmoidy, softmax, złożone straty).
- Szeroka społeczność; stabilne wersje pod Windows/Linux/CUDA.

### PyTorch Geometric (PyG)
**Rola:** heterogeniczny graf encji i faktów; message passing po krawędziach rolowych.

**Uzasadnienie:**
- Natywna obsługa `HeteroData` — wiele typów węzłów (encje, klastry-fakty) i wiele typów krawędzi (role ARG0/AGENT/THEME, reguły).
- Reifikacja faktów n-arnych jest wprost mapowalna na węzły zdarzeń z krawędziami ról.
- `MessagePassing` bazowa klasa pozwala wpiąć własne `Δs = p_u W⁺ - p_u W⁻` bez warstwy GCN/GAT.
- Obsługa sparse grafów; wydajna dla dużej liczby faktów.

**Alternatywa:** DGL — podobne możliwości, nieco inna ergonomia; wybierz PyG jeśli priorytetem jest zgodność z PyTorch Geometric Temporal lub ogólny ekosystem PyG.

### JAX + Equinox / Flax
**Rola:** opcjonalny zamiennik PyTorch dla wydajniejszego differentiate-through-loop.

**Uzasadnienie:**
- `jax.lax.scan` pozwala unrollować T kroków bez materializ. wszystkich gradientów (mniejsza pamięć niż BPTT w PyTorch).
- Warto rozważyć, jeśli T jest duże (>20 kroków) lub grafy są bardzo głębokie.
- Większy koszt wdrożenia — zalecane tylko jeśli profil wydajności tego wymaga.

---

## 2. Warstwa symboliczna — Symbolic Verifier

### Clingo (Python API: `clingo`)
**Rola:** wnioskowanie Horn+NAF (Answer Set Programming), stratyfikowana negacja, wyjątki `ab_*`.

**Uzasadnienie:**
- ASP z negacją jako `not p` naturalnie koduje NAF i wyjątki.
- Stratyfikacja negacji zapewnia deterministyczność (brak wieloznaczności).
- Provenance: Clingo zwraca atom-uzasadnienia; można je zmapować na DAG wyprowadzeń.
- Dojrzałe narzędzie (ponad 20 lat rozwoju), stabilne Python bindingi (`clingo` PyPI).
- Obsługuje `#minimize`, `#maximize` — przydatne do soft-rankingu kandydatów z proposera.

**Alternatywa:** `pyDatalog` — lżejszy Datalog w Pythonie; brak pełnego NAF. Dobry do prostych reguł bez wyjątków.

### NetworkX
**Rola:** budowa i przechowywanie DAG provenance (proof traces).

**Uzasadnienie:**
- DAG wyprowadzeń (kroki reguł + podstawienia) mapuje się wprost na `DiGraph`.
- Proste API do traversal, eksportu JSON, wizualizacji.
- Lekki — nie wymaga zależności GPU ani specjalistycznego środowiska.

---

## 3. Warstwa pamięci encji — Entity Memory

### FAISS
**Rola:** wyszukiwanie podobnych encji (entity linking) po embeddingach pamięci `m_e ∈ R^d`.

**Uzasadnienie:**
- Najszybsze approximate nearest-neighbor search dla wektorów; działa lokalnie bez serwera.
- Deterministyczny tryb `IndexFlatL2` (brak losowości) — spełnia wymóg deterministyczności.
- Skaluje się do milionów encji.

**Alternatywa:** `Annoy` — prostszy, mniejszy footprint; wystarczający jeśli liczba encji < 100k.

### SQLite (biblioteka standardowa + `sqlite-vec` lub `sqlite-vss`)
**Rola:** magazyn slotów encji (wartości symboliczne, wersje, provenance źródeł).

**Uzasadnienie:**
- Dane instancyjne (adresy, nazwy, identyfikatory, aliasy) pasują do relacyjnego modelu.
- SQLite jest deterministyczny, nie wymaga serwera, obsługuje wersjonowanie wierszy.
- Rozszerzenia `sqlite-vec` / `sqlite-vss` dokładają wyszukiwanie wektorowe — alternatywa dla FAISS jeśli chcesz jednej bazy.

---

## 4. NLP — ekstrakcja faktów z tekstu

### spaCy
**Rola:** pipeline ekstrakcji encji i relacji z tekstu; wstępne clampowanie faktów.

**Uzasadnienie:**
- Szybkie NER, dependency parsing, tokenizacja — podstawa do wypełniania slotów pamięci.
- Komponenty `EntityLinker` (do istniejącej bazy) i `RelationExtractor` można wpiąć w proposera.
- Deterministyczny pipeline (brak losowości w tokenizacji i NER przy stałych wagach).

### Hugging Face Transformers + `sentence-transformers`
**Rola:** generowanie embeddingów encji `m_e` i embeddingów kontekstowych faktów.

**Uzasadnienie:**
- `sentence-transformers` daje gotowe modele do podobieństwa semantycznego — używane w entity linking.
- Modele encoder (np. `roberta-base`, wielojęzyczne `xlm-roberta`) jako backbone proposera.
- Duży wybór pre-trained modeli; łatwa integracja z PyTorch.

---

## 5. Uczenie i eksperymenty

### PyTorch Lightning
**Rola:** strukturyzacja pętli treningu, checkpointing, logowanie.

**Uzasadnienie:**
- Oddziela logikę modelu od kodu treningu; ułatwia debugowanie wielokrokowego BPTT.
- Automatyczny mixed-precision (oszczędność pamięci przy unrollu T kroków).
- Nie wymagany — ale znacznie redukuje boilerplate.

### Weights & Biases (`wandb`) lub MLflow
**Rola:** śledzenie eksperymentów: hiperparametry T, λ, μ, β, sparsity wag, krzywe strat.

**Uzasadnienie:**
- Regularyzacja logiczna (L_imp, L_incomp, L_sparse) wymaga monitorowania wielu składowych straty jednocześnie.
- W&B daje wykresy, tabele, porównania runs; MLflow jest lokalny (bez chmury).

---

## 6. Podsumowanie — mapa bibliotek

| Warstwa / funkcja         | Biblioteka                         | Priorytet |
|---------------------------|------------------------------------|-----------|
| Framework neuronowy        | **PyTorch**                        | Krytyczny |
| Heterogeniczny graf/MP     | **PyTorch Geometric**              | Krytyczny |
| Symboliczny weryfikator    | **Clingo** (Python API)            | Krytyczny |
| Entity memory store        | **SQLite** + `sqlite-vec`          | Wysoki    |
| Entity linking (ANN)       | **FAISS**                          | Wysoki    |
| NLP / ekstrakcja faktów    | **spaCy**                          | Wysoki    |
| Embeddingi encji           | **sentence-transformers**          | Wysoki    |
| DAG provenance             | **NetworkX**                       | Średni    |
| Trening / boilerplate      | **PyTorch Lightning**              | Średni    |
| Śledzenie eksperymentów    | **W&B** lub **MLflow**             | Średni    |
| Wnioskowanie alternatywne  | `pyDatalog` (bez pełnego NAF)      | Opcjonalny|
| Wydajny unroll T kroków    | **JAX** + Equinox                  | Opcjonalny|
