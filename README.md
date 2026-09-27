# Lite-GD

Paper-faithful reproduction of Lite-GD for road-network carpool route planning.

This repository focuses on reproducing the published Lite-GD method with auditable data processing, leakage-free splits, exact route evaluation, and comparable baselines.

## Verified results

All methods below are evaluated on the same test split within each reconstructed benchmark.

| Method | Chengdu Gap ↓ | Chengdu Event-order Exact ↑ | Qingdao Gap ↓ | Qingdao Event-order Exact ↑ |
|---|---:|---:|---:|---:|
| DisGreedy | 9.883% | 54.30% | 6.274% | 73.66% |
| PointerNet | 9.269% | 68.90% | 4.281% | 74.63% |
| AM | 7.248% | 65.60% | 4.140% | 76.50% |
| Graph2Route | 5.445% | **72.60%** | 2.868% | **82.03%** |
| **Lite-GD (paper-faithful)** | **4.879%** | 68.30% | **2.698%** | 80.81% |

**Metrics.** Gap is the route-length gap to the exact optimum (lower is better). Event-order Exact requires the complete predicted pickup/drop-off event sequence to match the optimum (higher is better).

### Published Lite-GD reference

For context, the published Lite-GD table reports:

| Dataset | Reported Gap ↓ | Reported Accuracy ↑ |
|---|---:|---:|
| Chengdu | 4.85% | 80.26% |
| Qingdao | 6.54% | 84.04% |

The Chengdu reconstruction closely matches the published route-gap result. The Qingdao reproduction uses the recovered million-edge road graph with a link-midpoint adaptation and a group-safe split, so its absolute numbers should not be treated as a direct reproduction of the original Qingdao protocol.

## Benchmark protocols

### Chengdu

- Reconstructed paper-scale benchmark: 10,000 cases.
- Fixed split: 8,000 / 1,000 / 1,000 train/validation/test.
- Recovered code asset: 1,901 nodes / 5,941 directed edges.
- Exact route labels and route costs are independently certified.
- The retained source graph differs slightly from the paper table (1,902 nodes / 5,940 edges); the repository reports the recovered code asset rather than silently modifying it.

### Qingdao

- Recovered road graph: about 821k nodes and 2.13M directed edges.
- Largest strongly connected component is used.
- Candidate locations use the documented link-midpoint adaptation because the original point-on-link ratios are unavailable.
- Group-safe split prevents the same source order ID from leaking across train/validation/test.
- Lite-GD uses exact local receptive-field execution for scalable GNN inference; this is mathematically equivalent to the corresponding full-graph computation for the requested candidate embeddings.

## Repository layout

```text
Lite-GD/
├── repro/          # current reproduction, benchmarks and baselines
├── sim_data/       # retained Chengdu source assets
├── docs/
│   ├── REPRO_AUDIT.md
│   └── history/    # superseded intermediate reports
└── README.md
```

## Main implementation

- `repro/historical_full_model.py` — joint node/edge GNN and paper-faithful decoder.
- `repro/crossgraph_train.py` — Chengdu training/evaluation pipeline.
- `repro/qingdao_local_model.py` — exact local receptive-field execution for the million-edge Qingdao graph.
- `repro/qingdao_train.py` — Qingdao training/evaluation pipeline.
- `repro/paper_baselines.py` — Graph2Route-style and DisGreedy baselines.
- `repro/am_fidelity.py` — AM baseline.
- `repro/baseline_carpool.py` — PointerNet and common carpool baseline utilities.

## Reproduction policy

A result is promoted as reproduction evidence only when it has:

- no train/test leakage;
- a fixed split and random seed;
- exact or independently certified route labels;
- pickup-before-own-dropoff precedence;
- directed road-network evaluation;
- held-out test reporting;
- multi-seed confirmation where required for final Lite-GD results.

See [docs/REPRO_AUDIT.md](docs/REPRO_AUDIT.md) for source provenance, historical-code issues, and protocol details.

Legacy public code and vendored third-party repositories have been removed from the active tree and preserved separately in the project archive.
