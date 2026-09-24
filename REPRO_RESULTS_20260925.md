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
