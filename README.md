# Lite-GD

Official implementation of **Lite-GD** for road-network carpool route planning.

This repository contains the current Lite-GD implementation, benchmark construction and evaluation pipelines, together with the baseline methods used in our experiments.

## Latest results

The table below reports the latest results from the current implementation on the Chengdu and Qingdao benchmarks.

| Method | Chengdu Gap ↓ | Chengdu Event-order Exact ↑ | Qingdao Gap ↓ | Qingdao Event-order Exact ↑ |
|---|---:|---:|---:|---:|
| DisGreedy | 9.883% | 54.30% | 6.274% | 73.66% |
| PointerNet | 9.269% | 68.90% | 4.281% | 74.63% |
| AM | 7.248% | 65.60% | 4.140% | 76.50% |
| Graph2Route | 5.445% | **72.60%** | 2.868% | **82.03%** |
| **Lite-GD** | **4.879%** | 68.30% | **2.698%** | 80.81% |

**Metrics.** Gap is the route-length gap to the exact optimum (lower is better). Event-order Exact requires the complete predicted pickup/drop-off event sequence to match the optimum (higher is better).

## Benchmarks

### Chengdu

- 10,000 cases.
- Fixed split: 8,000 / 1,000 / 1,000 train/validation/test.
- Road network: 1,901 nodes and 5,941 directed edges.
- Directed road-network route costs and exact route labels.

### Qingdao

- Road network: about 821k nodes and 2.13M directed edges.
- Largest strongly connected component is used.
- Group-safe train/validation/test split.
- Local receptive-field execution is used for scalable GNN inference on the large road graph.

## Repository layout

```text
Lite-GD/
├── src/            # implementation, benchmarks and baselines
├── sim_data/       # Chengdu data assets
└── README.md
```

## Main implementation

- `src/historical_full_model.py` — joint node/edge GNN and Lite-GD decoder.
- `src/crossgraph_train.py` — Chengdu training and evaluation pipeline.
- `src/qingdao_local_model.py` — local receptive-field GNN execution for Qingdao.
- `src/qingdao_train.py` — Qingdao training and evaluation pipeline.
- `src/paper_baselines.py` — Graph2Route-style and DisGreedy baselines.
- `src/am_fidelity.py` — Attention Model baseline.
- `src/baseline_carpool.py` — PointerNet and common carpool baseline utilities.

## Evaluation setup

All methods in the latest-results table are evaluated with the same split and route-cost definition within each benchmark. The implementation enforces pickup-before-dropoff precedence and evaluates routes on directed road networks.
