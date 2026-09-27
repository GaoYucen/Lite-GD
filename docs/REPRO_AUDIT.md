# Lite-GD Reproduction Audit

Date: 2026-09-25  
Branch: `repro/paper-aligned-20260925`

## 1. Scope

This branch does **not** treat the current repository result as a reproduction of the WWW 2024 / TMC 2025 Lite-GD model.

The goals are:

1. audit the published method against the checked-in implementation;
2. recover deleted historical Chengdu assets where possible;
3. establish leakage-free train/validation/test evaluation;
4. rebuild paper-aligned components in independently testable stages;
5. only claim a result after it is reproducible from a fixed split and seed.

The original `main` branch is left unchanged.

## 2. Main finding

The documented entrypoint `code/Lite-GD/train.py` trains `PointerNet.py`, not the paper's full Lite-GD model.

The paper describes the following chain:

`node+edge input embedding -> lightweight/filter-integrated GCN -> supervised node/edge pre-training -> candidate distance/angle feature crossover -> gating -> sequential attention decoder -> rule-based mask`.

The checked-in main training path does not implement that chain.

`MCRP_Net.py` contains a partial GCN/decoder prototype, but it is not the main training path and cannot be considered a faithful implementation.

## 3. Concrete implementation problems

### 3.1 Missing runtime assets

The current repository does not contain files referenced by the documented code:

- `sim_data/chengdu_data.npy`
- `sim_data/chengdu_directed_shortest_distance_matrix.npy`
- `sim_data/Data_Generator.py`
- `distance_matrix.pth`
- `chengdu_case_feature.txt`
- `chengdu_order_label.txt`

The documented training entrypoint currently fails on `Data_Generator` before training begins.

### 3.2 Data leakage

The legacy training path loads the full `chengdu_data.npy` as the training dataset.  
The legacy validation path then loads the same full file again.

There is no 8:1:1 split in the checked-in training/evaluation path, even though the paper reports an 8:1:1 split.

The legacy "best" checkpoint is selected by training loss, not independent validation loss.

### 3.3 Decoder constraint mismatch

The paper's rule mask is a precedence mask:

- driver first;
- one candidate per candidate set;
- passenger drop-off is unavailable until that passenger's pickup has been visited.

This allows legal interleaving, e.g. pickup-1 -> dropoff-1 -> pickup-2 -> dropoff-2.

The legacy PointerNet instead hard-codes the first two decisions as pickup groups and the last two as drop-off groups. It therefore solves a restricted pickup-first subproblem.

It also contains an off-by-one mask slice (`11:-1`) that leaves the last candidate unmasked.

### 3.4 Missing paper components

The main training path has no:

- node+edge GCN encoder used by the published decoder;
- supervised complete-route node/edge pre-training task;
- binary / variance filters;
- spherical-distance + direction-angle feature crossover;
- published gating equations.

### 3.5 Problems in `MCRP_Net.py`

The prototype contains several reproducibility/correctness issues:

- hard-coded local paths under `/Users/mali/...`;
- missing DGL/runtime data;
- hard-coded `nn.Linear(5941, 32)`;
- global mutable objects;
- the same graph object is reused for multiple cases;
- case-specific edge update uses `edges_of_id.iloc[0]` for every edge rather than the current row;
- candidate state can be chosen before the mask is correctly applied;
- pickup-first decoder logic is again hard-coded;
- no faithful distance+angle crossover/gating implementation;
- no faithful pre-training loop or published filters.

## 4. Dataset audit

### Published Chengdu benchmark

The papers report:

- 5,940 edges
- 1,902 nodes
- 10,000 samples
- 8:1:1 train/validation/test split

### Current checked-in simplified data

The checked-in graph/data audit gives:

- 5,941 road-edge rows
- 1,901 node rows
- 1,000 simplified cases
- two passengers only
- 5 candidates per event group
- candidates represented by graph nodes rather than the paper's edge+ratio point representation

The simplified optimal sequence labels contain only four pickup-first patterns. The two additional legal interleaving patterns permitted by the paper's precedence definition never occur.

### Recovered historical edge-level data

Git history contains deleted files much closer to the paper formulation:

- `chengdu_link_feature.txt`
- `chengdu_order_1000.txt`
- `chengdu_label_1000.txt`
- `chengdu_data.npy`

The edge-level files contain candidate edge IDs, candidate ratios, optimal selected candidate edges, complete optimal node routes, complete optimal edge routes, and route lengths.

They are sufficient to reconstruct most of the uncommitted `chengdu_case_feature.txt` semantics and the paper pre-training labels.

However, this historical set is still not the final published Chengdu benchmark:

- 1,000 cases, not 10,000;
- 950 two-passenger + 50 three-passenger cases;
- all 1,000 optimal event sequences are pickup-first;
- average route length differs substantially from the published benchmark.

Therefore it should be treated as a historical development dataset, not as evidence for Table VI.

## 5. Legacy checkpoint diagnostic

The deleted `chengdu_data.npy` was recovered from Git history.

Each sample contains:

- `Points_id`: 21 graph nodes;
- `Points`: 21 x 23 feature matrix;
- `Solutions`: pointer indices;
- `Opt_Seq`;
- `Opt_Length`.

The 23 input dimensions are two coordinates plus 21 distance-related features.

A correct directed-road all-pairs shortest-path matrix was reconstructed from the checked-in Chengdu graph. It matches stored optimal route lengths with approximately numerical floating-point error only.

Using that road metric, the legacy checkpoint evaluated on the same 1,000 cases used by the legacy training path gives:

- average optimal length: 30,894.71
- average predicted length: 31,240.83
- route gap: 1.12%
- exact pointer sequence accuracy: 67.1%

This is a same-data diagnostic, **not** a valid generalization result. It also does not reproduce the published Chengdu 4.85% gap / 80.26% accuracy result and is not directly comparable because the benchmark is different.

## 6. Clean reproduction layer

### `repro/strict_pointer.py`

Purpose: controlled diagnostic baseline, **not** claimed as Lite-GD.

Properties:

- deterministic 8:1:1 split;
- independent validation-only checkpoint selection;
- untouched test split;
- canonical repair of 16 archived pointer-index ambiguities caused by repeated graph nodes;
- paper precedence mask;
- directed-road route-length evaluation;
- coordinate-only and archived distance-feature ablations.

### `repro/historical_chengdu.py`

Purpose: deterministically reconstruct historical edge-level cases and paper-style pre-training labels from deleted Git assets.

### `repro/litegd_core.py`

Purpose: test a paper-aligned online core on the simplified archived data:

- 2-layer road-topology GCN;
- spherical-distance features;
- direction-angle features;
- Hadamard feature crossover;
- published two-gate fusion structure;
- sequential attention decoder;
- paper precedence/group mask.

It is deliberately not called a full reproduction because the simplified 1,000-case tensor does not contain the complete edge+ratio / complete-route supervision needed for the paper's pre-training and filter stages.

## 7. Reproduction strategy

The recommended sequence is:

1. **Audit baseline**
   - finish corrected strict 8:1:1 multi-seed evaluation;
   - report mean/std, not one seed.

2. **Historical edge-level reconstruction**
   - recover candidate edge/ratio semantics;
   - reconstruct optimal-route node/edge labels;
   - validate all records before training.

3. **Paper core**
   - node and edge embeddings;
   - 2-layer GCN;
   - distance/angle feature crossover;
   - gating;
   - rule-based sequential decoder.

4. **Pre-training**
   - node binary classification from complete optimal route;
   - edge four-class classification from candidate set M and optimal route O;
   - downstream fine-tuning.

5. **Filters**
   - add binary and variance filters only after the unfiltered model is correct;
   - document every approximation where the paper is underspecified.

6. **Fresh Chengdu benchmark**
   - generate a new 10,000-case synthetic dataset from the road graph with a fixed seed;
   - use edge+ratio candidate points;
   - compute exact labels using directed shortest paths + exact combinatorial search;
   - include legal interleaving routes rather than enforcing pickup-first.

7. **Final evidence**
   - >= 3 seeds for development, preferably 5 seeds for final tables;
   - PointerNet / AM-style / DisGreedy baselines on exactly the same split;
   - component ablations for pre-training, crossover, GCN and filters;
   - route gap, exact sequence accuracy, query time and training time.

## 8. What counts as success

A trustworthy reproduction should satisfy all of the following:

- no train/test leakage;
- exact data split and seed recorded;
- route metric independently validated against exact labels;
- paper mask semantics implemented;
- component ablations show reproducible direction of effect;
- result stable across seeds;
- no claim of matching the published table unless the benchmark itself is actually equivalent.



## 9. Exact-label result snapshot

Detailed numbers are maintained in [REPRO_RESULTS_20260925.md](REPRO_RESULTS_20260925.md).

After replacing the archived pickup-first targets with exact paper-precedence labels, the current three-seed online core reaches **4.68% mean route gap** with GCN + distance/angle feature crossover + gating. This is a real held-out improvement over the strict archived PointerNet (5.55%) and nearest-feasible greedy (6.04%), but the exact-sequence accuracy remains low (~5.33%) and the retained 1,000-case benchmark is not the paper's 10,000-case Chengdu benchmark.

The next scientific gate is therefore the paper's node/edge supervised pre-training on recovered edge+ratio cases, not additional tuning of the simplified PointerNet.


## 10. Current fixed-split and cross-graph protocol

The current reproduction stage intentionally separates **model comparison** from
**graph-scale validation**.

### 10.1 Fixed-split learning comparison

The historical edge+ratio benchmark now uses one frozen split:

- retained cases: 996;
- split seed: `20260925`;
- training seeds: `1234, 4321, 2468`;
- primary metrics:
  - Mean-case Gap;
  - Event-step Accuracy.

The same train/validation/test case IDs are used for PointerNet, AM-style and
Lite-GD.  Detailed numbers are in `REPRO_RESULTS_20260925.md`.

### 10.2 Cross-graph benchmark policy

The first graph-scale experiment fixes the routing-task complexity at **two
passengers** so road-network scale is the principal changed variable.

Recovered Chengdu task statistics used to calibrate generation:

- candidate count per semantic event:
  - range 4--10;
  - mean 7.56;
  - median 8;
- candidate-group radius (projected/geodesic metres):
  - median about 490 m;
  - p90 about 1.17 km;
  - p95 about 1.71 km;
- driver -> exact pickup directed-road distance:
  - median about 8.96 km;
- exact pickup -> own dropoff directed-road distance:
  - median about 9.93 km;
- historical candidate edge-ratio semantics:
  - pickups near 0.001;
  - dropoffs near 0.999.

Generated cases use:

1. one driver point on a directed road edge;
2. two passenger pickup/dropoff event pairs;
3. variable-size candidate groups calibrated to the recovered Chengdu count and
   spatial-scale distributions;
4. pickup ratio 0.001 and dropoff ratio 0.999;
5. each pickup constrained to precede its own dropoff;
6. exact directed point-on-edge distance;
7. exact dynamic-programming search over all legal event orders and candidate
   choices;
8. deterministic 8:1:1 split with split seed `20260925`.

Synthetic event centres are first **snapped to valid road support** before local
candidate sets are drawn.  This avoids inflating candidate-group radius when a
Euclidean target falls inside a road-sparse block.

### 10.3 Reused audited road assets

No new OSM download is required.

| Graph | Nodes | Directed arcs | Exact-distance asset |
|---|---:|---:|---|
| Chengdu | 1,901 | 5,941 | existing APSP |
| Jinan | 8,840 | 23,206 | Lite-GD cross-graph APSP, generated once from audited native graph |
| Shenzhen | 11,738 | 27,105 | existing audited APSP |
| DIMACS-FLA | 1,070,376 | 2,687,902 | existing RoutingKit CH exact oracle |

Jinan/Shenzhen first undergo a 1,000-case distribution-certification pilot.
Only after the candidate/OD distributions are accepted is the identical
generator frozen for the 10,000-case benchmark.  FLA remains a later stage
because the full Lite-GD reproduction requires recovering the paper filter /
route-supervision path without constructing an impossible million-node APSP.
