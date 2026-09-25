# Lite-GD Reproduction Results — 2026-09-25

Branch: `repro/paper-aligned-20260925`

This document records the first **leakage-free, exact-label** reproduction evidence obtained during the 2026-09-25 audit. The original `main` branch is intentionally unchanged.

## 1. Why the legacy result is not trustworthy

The historical checkpoint `code/Lite-GD/param/param_5_1000_best.pkl` looks strong when evaluated on the deleted 1,000-case tensor:

- mean route gap: **1.17%**
- exact pointer-sequence accuracy: **67.1%**
- selected-point accuracy: **88.95%**

However, legacy `train.py` and `val.py` both load the same `chengdu_data.npy`, with no independent 8:1:1 train/validation/test split. The training script also reloads an existing checkpoint. These numbers are therefore a same-data diagnostic, not a generalization result.

## 2. The archived labels solve a narrower problem than the paper

The paper rule is per-passenger precedence:

> pickup(i) must precede drop-off(i).

For two passengers this permits **6** legal event-group orders, including interleaved orders such as pickup-1 → drop-off-1 → pickup-2 → drop-off-2.

The archived generator instead enforces all pickups before all drop-offs. Exact directed-road search over all six paper-legal orders shows:

| Exact-label audit | Value |
|---|---:|
| Cases | 1,000 |
| Legacy labels that are also true paper optimum | **596 (59.6%)** |
| Cases improved by allowing paper-legal interleaving | **404** |
| Legacy interleaved cases | **0** |
| Exact interleaved cases | **404** |
| Pointer sequence changed after exact relabel | **418** |
| Mean legacy-label gap vs exact paper OPT | **4.71%** |
| 95th-percentile legacy-label gap | **25.64%** |
| Maximum legacy-label gap | **72.71%** |

The stored legacy route lengths themselves are internally consistent with the reconstructed directed-road metric (mean absolute recalculation error ~0.00067), so the main problem is the **search-space restriction**, not a floating-point/path-length bug.

The corrected exact dataset is generated at:

`/workspace/.server-control/litegd-paper-data/chengdu_data_paper_exact.npy`

and is reproducible with `repro/relabel_exact_opt.py`.

## 3. Strict 8:1:1 evaluation protocol

All results below use:

- deterministic 8:1:1 split;
- independent validation checkpoint selection;
- untouched test set;
- three seeds: 20260925, 20260926, 20260927;
- directed road-network route length;
- paper precedence mask;
- no illegal predicted routes.

The metric `Gap` is the ratio of average predicted route length to average exact-optimal route length, matching the paper's average-length comparison.

## 4. Exact-label results

### 4.1 Learning methods

| Model | Mean Gap | Std | Mean exact-sequence Acc | Mean pointer Acc |
|---|---:|---:|---:|---:|
| Coord-only decoder | 8.04% | 0.20 | 3.00% | 33.08% |
| GCN-only core | 7.55% | 0.27 | 5.33% | 35.33% |
| Legacy 23D PointerNet, paper mask | 5.55% | 0.88 | 3.67% | 36.92% |
| Domain crossover only | 5.15% | 1.11 | 4.67% | 42.08% |
| **GCN + domain crossover + gating** | **4.68%** | **1.25** | **5.33%** | **42.92%** |

Per-seed gaps for GCN + domain crossover + gating:

- 20260925: **3.24%**
- 20260926: **5.54%**
- 20260927: **5.25%**

The direction is important: after fixing the target labels, the full online core becomes better than either GCN-only or domain-only. This is the first credible evidence in this repository that the paper's two information sources are complementary rather than decorative.

### 4.2 Routing/search references

| Baseline | Mean Gap across 3 splits | Interpretation |
|---|---:|---|
| **Restricted pickup-first exact search** | **3.74%** | Exact optimizer in the old, narrower search space |
| Nearest feasible greedy | 6.04% | Simple paper-legal greedy |
| Legacy 23D PointerNet | 5.55% | Leakage-free learned baseline |
| GCN + domain + gating | 4.68% | Current paper-core learned model |
| Full paper-precedence exact search | 0.00% | Ground-truth oracle |

The archived legacy labels and restricted pickup-first exact search are numerically the same on these cases. The restricted exact oracle is therefore a useful correctness reference, but it is not a like-for-like learned baseline: it explicitly enumerates the reduced combinatorial search space.

The current learned core beats nearest-feasible greedy and the strict PointerNet, but does **not yet** beat the restricted exact oracle.

## 5. Relation to the TMC paper

TMC Table VI reports on its **10,000-case Chengdu benchmark**:

- Lite-GD gap: **4.85%**
- exact sequence accuracy: **80.26%**

Our corrected 1,000-case archived benchmark gives:

- GCN + domain + gating mean gap: **4.68%**
- exact sequence accuracy: **5.33%**

The similar gap is encouraging, but it must **not** be described as a reproduction of Table VI:

1. this repository only retains 1,000 simplified node-candidate cases, not the paper's 10,000 cases;
2. the paper uses edge+ratio candidate points and a joint node-edge encoder;
3. the current core has no supervised node/edge pre-training and no filters;
4. exact-sequence accuracy is far below the paper result.

The correct conclusion is:

> **The paper's core online design has now shown real held-out effectiveness on a corrected benchmark, but the full published Lite-GD result has not yet been reproduced.**

## 6. What the experiments say about the architecture

### GCN is not useless

On the old restricted labels, GCN-only improved the three-seed held-out gap from approximately 4.43% (coordinate core) to approximately 3.25%.

On corrected exact labels, GCN-only is weaker by itself (7.55%), but combining GCN with domain features improves domain-only from 5.15% to 4.68%.

Thus the role of GCN changes with the target distribution, but it provides complementary structural information in the corrected problem.

### Domain feature crossover is important

On exact labels, adding the distance/angle domain pathway dramatically improves over coordinate-only and GCN-only representations. This is directionally consistent with the TMC ablation, where removing feature crossover causes a large degradation.

### The biggest remaining gap is candidate discrimination

Route-length gap is already reasonable, while exact-sequence accuracy remains very low. This pattern suggests many predicted routes are near-optimal but do not select exactly the same candidate sequence as the oracle.

The next component to test should therefore be the paper's **supervised node/edge pre-training**, whose explicit purpose is to distinguish sparse candidate-relevant road elements.

## 7. Next reproduction stage

The full pre-training stage should not be faked on the simplified node-only tensor.

The recovered historical data contains the information needed for a more faithful encoder:

- candidate edge IDs;
- edge ratios;
- event types;
- full road graph;
- complete historical route paths.

The next stage is:

1. regenerate paper-precedence exact labels for the historical edge+ratio cases under one consistent directed-road metric;
2. reconstruct exact complete node/edge routes;
3. build case-conditioned edge inputs: start/end coordinates, edge length, candidate ratio, candidate/event type;
4. implement joint node-edge GCN;
5. pre-train:
   - node binary route-membership classification;
   - edge four-class classification based on candidate set M and exact route O;
6. fine-tune the sequential decoder;
7. evaluate at least 3 seeds against the exact-label no-pretraining core;
8. add binary/variance filters only after pre-training is verified.

A pre-training component is retained only if its held-out improvement is stable across seeds.

## 8. Current scientific status

**Verified:**

- old public implementation is not the paper model;
- old headline checkpoint metric is leakage-prone;
- old labels enforce an incorrect narrower ordering constraint;
- exact paper-precedence labels can be regenerated deterministically;
- GCN, domain crossover, gating and paper mask can produce a real held-out improvement;
- the combined paper core reaches **4.68% mean gap** over three seeds on corrected exact labels.

**Not yet verified:**

- full node+edge pre-training gain;
- filter gain/efficiency trade-off;
- paper's 80.26% Chengdu exact accuracy;
- the published 10,000-case Chengdu table;
- full WWW/TMC figures and efficiency table.



## 9. Historical edge+ratio exact-label audit

The older edge-level development set is closer to the WWW/TMC formulation than the simplified 21-node tensor:

- 1,000 cases;
- 950 two-passenger cases;
- 50 three-passenger cases;
- candidate edge IDs and edge ratios;
- complete historical node/edge routes.

It is usable as an **input source**, but its historical labels are not reliable ground truth.

A paper-precedence exact solver was run directly on the recovered edge+ratio candidate points with one unified directed point-on-edge metric. The old selected sequences were then re-evaluated under exactly the same metric, rather than comparing against the historical stored `route_length`.

| Historical edge+ratio audit | Value |
|---|---:|
| Cases | 1,000 |
| Old selected sequences that remain exact-optimal | **130** |
| Mean old-solution gap vs exact OPT | **21.32%** |
| Median gap | **17.07%** |
| 95th-percentile gap | **57.68%** |
| Maximum gap | **142.44%** |
| Negative-gap cases below tolerance | **0** |
| Exact solutions using interleaved pickup/drop order | **420** |
| Cases whose selected candidate sequence changes | **888** |

There are 342 occurrences where an old selected edge ID appears more than once inside the same semantic event's candidate list. All 342 duplicates have the **same candidate ratio**, so the selected road point is still unique geometrically; none of the 21.32% gap is caused by choosing the wrong ratio for a duplicate edge ID.

This audit strengthens the main conclusion:

> Historical edge-level files should be used to reconstruct inputs and supervision structure, but their saved “optimal” labels should not be used for training or evaluation.

The regenerated exact edge+ratio cases are the basis for the next node/edge pre-training experiment.

## 10. Pre-training implementation gate

Before testing the paper's pre-training, the joint encoder was corrected to make node states **case-conditioned**:

`edge static features + candidate event type + candidate ratio -> edge embedding -> incident-node aggregation -> node update -> endpoint attention -> edge update`.

This is necessary because route-membership node labels are case-specific. A static coordinate-only node GCN cannot learn a different node label for different MCRP cases.

The pre-training pilot compares the same joint GCN + distance/angle crossover + gating + precedence decoder:

- from scratch; versus
- initialized by exact-route node binary classification + edge four-class classification.

The pilot is retained only if held-out route quality improves, after which it will be repeated over multiple seeds.


## 11. Final historical edge+ratio pre-training result (3 seeds)

The final experiment uses commit `2dccc4d8f7634c69b7063a707f1afad412b219e3`, excludes the four historical cases whose passenger/group semantics cannot be uniquely reconstructed (`51, 244, 620, 873`), and evaluates **996 cases** with deterministic 8:1:1 splits.

The same joint node/edge GCN + domain crossover + gating + precedence decoder is trained:

- from scratch; and
- after exact-route node binary + edge four-class pre-training.

| Seed | Scratch Gap | Pretrained Gap | Gap change | Scratch Exact | Pretrained Exact | Scratch Pointer | Pretrained Pointer |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260925 | 6.70% | 6.60% | **-0.10pt** | 3.0% | 2.0% | 32.93% | 30.98% |
| 20260926 | 5.69% | 5.78% | +0.08pt | 1.0% | 3.0% | 36.34% | 40.49% |
| 20260927 | 6.81% | 5.60% | **-1.21pt** | 2.0% | 1.0% | 30.49% | 30.98% |
| **Mean** | **6.40%** | **5.99%** | **-0.41pt** | **2.0%** | **2.0%** | **33.25%** | **34.15%** |

Across seeds:

- scratch gap mean/std: **6.40% / 0.50%**;
- pretrained gap mean/std: **5.99% / 0.43%**;
- mean absolute gap reduction: **0.41 percentage points**;
- mean relative gap reduction: **6.38%**;
- mean-case gap: **6.14% -> 5.72%**;
- illegal-route rate: **0% for every run**.

Interpretation:

1. **The pre-training signal is real but modest.** Route quality improves on average and in two of three seeds.
2. **The gain is not yet robust enough to promote pre-training as a reproduced paper component.** One seed regresses slightly.
3. **Exact sequence discrimination remains the main failure mode.** Mean exact accuracy stays at 2%, while pointer accuracy improves by <1 point.
4. **Pre-training appears to shape the representation toward lower-cost near-optimal routes rather than toward exact candidate recovery.** This is useful, but it is not sufficient to explain the TMC exact-sequence result.

The next experiment therefore measures node/edge pre-training validation quality directly and tests whether pre-trained representations are being overwritten during decoder fine-tuning (lower encoder LR / staged unfreezing) before introducing the paper filters.


## 12. Pre-training mechanism diagnostics

Two representative seeds were rerun with a controlled diagnostic protocol. Pre-training is performed once, then the same pre-trained state is cloned into:

- normal fine-tuning (encoder and decoder share the same LR);
- a lower-encoder-LR fine-tuning variant.

This diagnostic uses a fresh deterministic fine-tuning DataLoader order, so its absolute route gaps are mechanism tests rather than replacements for the final 3-seed table in Section 11.

### 12.1 Pre-training classification quality

At epoch 8, validation metrics are:

| Metric | Seed 20260926 | Seed 20260927 |
|---|---:|---:|
| Node positive prevalence | 4.75% | 4.41% |
| Node positive F1 | **19.82%** | **16.81%** |
| Edge raw accuracy | 97.49% | 97.47% |
| Edge macro-F1 | **57.24%** | **57.51%** |
| Candidate-only F1 | 90.25% | 89.89% |
| Neither F1 | 98.76% | 98.75% |
| Route-only F1 | **11.01%** | **11.87%** |
| Candidate-and-route F1 | **28.96%** | **29.51%** |

Raw edge accuracy is misleading because the validation graph contains roughly 582k “neither” edges, versus only ~8–9k route-only edges and ~0.6k candidate-and-route edges.

The key failure mode is therefore explicit:

> the current pre-training learns candidate membership/background well, but learns exact-route relevance poorly.

This explains why route gap can improve while exact candidate-sequence recovery remains weak.

### 12.2 Fine-tuning does not appear to overwrite a good pre-trained encoder

Controlled diagnostic route gaps:

| Seed | Scratch | Pretrain, same LR | Pretrain, lower encoder LR |
|---|---:|---:|---:|
| 20260926 | 5.96% | **4.79%** | 6.89% |
| 20260927 | 6.45% | **5.80%** | 6.75% |

Lowering the encoder LR makes both seeds worse. The main bottleneck is therefore **not** catastrophic overwriting during decoder fine-tuning; the pre-training signal itself needs to become more route-discriminative.

### 12.3 Two-passenger and three-passenger behavior differ sharply

Seed 20260926:

| Strategy | 2-passenger Gap (95 test cases) | 3-passenger Gap (5 test cases) |
|---|---:|---:|
| Scratch | 4.08% | 32.82% |
| Pretrain, same LR | **3.84%** | **18.37%** |
| Pretrain, lower encoder LR | 4.41% | 42.35% |

Seed 20260927:

| Strategy | 2-passenger Gap (95 test cases) | 3-passenger Gap (5 test cases) |
|---|---:|---:|
| Scratch | 5.25% | 23.88% |
| Pretrain, same LR | **4.22%** | 28.67% |
| Pretrain, lower encoder LR | 5.30% | 27.67% |

For two passengers, normal pre-training improves both representative seeds. Three-passenger behavior is unstable and much worse in absolute quality.

After excluding four ambiguous two-passenger cases, the retained dataset has 946 two-passenger and only 50 three-passenger cases. The stratified 8:1:1 split therefore gives only about **40 three-passenger training cases and 5 three-passenger test cases per seed**. A one-case change is 20% of the three-passenger test set.

### 12.4 Next optimization gate

The next experiment keeps the paper four-class edge objective but adds:

1. a route/non-route auxiliary loss derived from the same four labels, to explicitly penalize:
   - route-only -> neither;
   - candidate-and-route -> candidate-only;
2. a moderate 5x repeat of three-passenger training cases, with validation/test untouched.

These are tested separately and together before any filter or larger model is introduced.


## 13. Route-aware / three-passenger balancing: rejected as the default

Mechanism diagnostics showed that sparse exact-route edge classes are difficult to learn and that the three-passenger subset is extremely small. Two targeted interventions were tested:

1. an auxiliary route-vs-non-route edge loss;
2. repeating three-passenger training cases 5× while leaving validation/test untouched.

Representative held-out results:

| Seed | Variant | 2p Gap | 3p Gap | Overall Gap |
|---|---|---:|---:|---:|
| 20260926 | route auxiliary | 4.60% | 44.76% | 7.22% |
| 20260926 | route auxiliary + 3p repeat5 | 4.95% | 24.87% | 6.26% |
| 20260927 | route auxiliary | 3.81% | 31.44% | 5.59% |
| 20260927 | route auxiliary + 3p repeat5 | 6.50% | 25.96% | 7.75% |

Repeating three-passenger cases does reduce the three-passenger gap relative to the corresponding route-auxiliary run, but it consistently trades away two-passenger quality and is unstable overall. Because the final test split contains only about five three-passenger cases, a one-case change is already a 20-point change in three-passenger exact accuracy.

**Decision:** do not use mechanical 5× three-passenger oversampling as the unified Lite-GD default. If three-passenger specialization is needed later, treat it as a separate low-data/generalization problem.

## 14. Candidate-aware pre-training: current best historical edge+ratio variant

The pre-training diagnostics showed a clear class-imbalance bottleneck:

- event ordering after fine-tuning is already fairly accurate;
- exact-route/candidate discrimination is much weaker;
- the four-class edge head is dominated by the `neither` class.

A candidate-aware auxiliary objective was therefore added **without changing the decoder or the train/validation/test splits**. It keeps the paper node/edge objectives and adds a binary loss only on candidate edges:

`candidate-only vs candidate-and-exact-route`.

With candidate-positive weight 3, validation `candidate_and_route` F1 increased from roughly 29% under ordinary pre-training to:

- **42.25%** on seed 20260925;
- **37.50%** on seed 20260926;
- **44.67%** on seed 20260927.

Final held-out route results:

| Seed | Standard pre-train Gap | Candidate-aware Gap | Improvement | 2p Gap | 3p Gap |
|---|---:|---:|---:|---:|---:|
| 20260925 | 6.60% | **5.71%** | **-0.89pt** | 4.57% | 20.40% |
| 20260926 | 5.78% | **5.16%** | **-0.62pt** | 3.59% | 27.58% |
| 20260927 | 5.60% | **5.38%** | **-0.22pt** | 3.83% | 27.79% |
| **Mean** | **5.99%** | **5.42%** | **-0.58pt** | **4.00%** | **25.26%** |

Across the three seeds:

- candidate-aware gap mean/std: **5.42% / 0.23%**;
- ordinary pre-trained gap mean/std: **5.99% / 0.43%**;
- candidate-aware improves every seed;
- mean absolute reduction vs ordinary pre-training: **0.58 percentage points**;
- mean relative reduction vs ordinary pre-training: about **9.4%**;
- scratch historical full-model gap was about **6.40%**, so the total reduction from scratch is about **0.98 points**.

### Sequence-error decomposition

The candidate-aware decoder exposes why exact sequence accuracy remains low.

Three-seed means:

- event-order exact accuracy: **72.0%**;
- event-order step accuracy: **83.9%**;
- candidate-by-event accuracy: **42.0%**;
- complete exact sequence accuracy: **1.33%**;
- pointer-position accuracy: **35.3%**;
- illegal route rate: **0%**.

Near-optimal route quality is much stronger than exact-sequence identity:

- within 1% of OPT: about **11.3%**;
- within 3%: about **46.0%**;
- within 5%: about **67.0%**;
- within 10%: about **89.7%**.

This leads to a concrete diagnosis:

> **The current model mostly understands the legal event order, but it still struggles to choose the exact road point inside each semantic pickup/drop-off candidate set.**

Therefore, the next useful work should target candidate discrimination or candidate-set scoring directly. More global decoder complexity or mechanical three-passenger oversampling is not the current priority.

### Current preferred historical edge+ratio configuration

For the recovered historical benchmark, the current preferred model is:

- paper-precedence exact relabeling;
- four ambiguous historical cases excluded;
- case-conditioned joint node/edge GCN;
- original node/edge pre-training objectives;
- candidate-aware auxiliary edge loss;
- distance + direction-angle crossover;
- gating;
- rule-constrained pointer decoder.

This configuration is currently the most stable full-model reconstruction on the available edge+ratio data. It is still **not** a reproduction of the published 10k benchmark because that benchmark is absent from the repository history.
