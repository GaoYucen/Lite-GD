# Paper-aligned Lite-GD reproduction

This directory is a clean reproduction track for the WWW 2024 / TMC 2025 Lite-GD papers. It is intentionally separated from the legacy entry points under code/Lite-GD.

## Why a separate track is needed

The repository main branch contains several historical prototypes rather than one reproducible paper implementation.

- code/Lite-GD/train.py trains a Pointer Network and does not call MCRP_Net.py.
- The documented training entry point currently depends on files that are absent from main, including chengdu_data.npy and the directed shortest-distance matrix.
- The historical Pointer Network training and validation scripts read the same chengdu_data.npy file, so the stored checkpoint cannot be treated as a held-out reproduction result.
- MCRP_Net.py contains a partial GCN prototype, but it has hard-coded local paths and does not implement the complete paper pipeline (supervised node/edge pre-training, domain feature crossover with angle features, gating, and the paper rule mask).
- Historical Git data contains a more paper-aligned edge-based Chengdu dataset that was deleted from main in November 2023. It will be recovered by commit SHA for forensic reproduction rather than copied silently into the current data directory.

The legacy files are retained unchanged for provenance.

## Reproduction stages

### Stage A — trustworthy held-out lower bound

heldout_pointer.py recovers the historical 1000-case chengdu_data.npy from Git history and applies a fixed 800/100/100 train/validation/test split.

Before training, it independently verifies each stored Opt_Length against exhaustive enumeration of the legal two-passenger candidate space. Training is from scratch. Model selection uses validation data only. Test metrics are reported once on the untouched test split.

The decoder uses a hard legality mask:

- the car is the fixed start point;
- exactly one candidate is selected from each of the four candidate sets;
- passenger 1 pickup must precede passenger 1 drop-off;
- passenger 2 pickup must precede passenger 2 drop-off.

Two variants are currently included:

- base: the recovered 23-dimensional historical point features;
- angle: base features plus sine/cosine direction-angle features to each candidate-set center, approximating the paper domain feature crossover signal.

This stage is not claimed to be the full paper architecture. Its purpose is to establish a leakage-free learning baseline and test whether the historical task itself supports genuine generalization.

### Stage B — paper-aligned edge/graph reconstruction

The next stage uses the deleted historical files at commit d8f8d5bc00ad1c605adf1c73c92c815cf12c7065:

- chengdu_order_1000.txt
- chengdu_label_1000.txt
- chengdu_link.txt
- chengdu_link_feature.txt
- chengdu_node.txt
- R1-link.csv

These files retain candidate edge IDs and edge ratios as well as the complete optimal node/edge route, making the supervised node/edge pre-training labels in the papers reconstructible.

The first paper-aligned graph experiment will use the 950 two-passenger cases. The final 50 cases contain six candidate sets (three passengers) but their ratio pattern is inconsistent with the semantic pickup/drop-off labels and is therefore kept out of the first certified run until repaired.

Planned graph variants:

1. node/edge GCN without filters;
2. + supervised node/edge pre-training;
3. + spherical-distance and angle feature crossover;
4. + gated fusion of crossover features and edge embeddings;
5. optional binary/variance filters after correctness is established.

The paper ablation indicates that feature crossover and pre-training are the large performance contributors, while filtering primarily targets scalability. Therefore correctness work is ordered accordingly.

## Metrics

Every certified run reports at least:

- mean / median / 95th-percentile route gap to exact OPT;
- exact route-sequence accuracy;
- selected-point accuracy;
- invalid-route rate;
- fixed split indices and random seed;
- Git commit SHA and environment information.

A result is not considered a reproduction result if it is evaluated on the training data or if its target OPT cannot be independently verified.
