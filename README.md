# Lite-GD

Paper-faithful reproduction of Lite-GD for road-network carpool route planning.

## Branches

- **main** — conference/reproduction line. It contains the paper-faithful Lite-GD model, reconstructed benchmark pipeline, baselines, and mathematically equivalent scalable execution backends.
- **journal/full** — research branch containing all subsequent experimental extensions, including directed road-metric relational modeling and hierarchical event-candidate decoding.

Legacy public code and historical branches are preserved in the private `GaoYucen/research-paper-archives` repository rather than kept in active branches.

## Current verified reproduction

| Dataset | Gap ↓ | Event-order Exact ↑ |
|---|---:|---:|
| Chengdu reconstructed 10k | **4.879%** | **68.30%** |
| Qingdao group-safe | **2.698%** | **80.81%** |

The Chengdu gap closely matches the reported paper result (4.85%). Qingdao uses the recovered million-edge road graph with the documented midpoint adaptation and group-safe split.

## Repository layout

```text
Lite-GD/
├── repro/          # current reproduction, benchmarks and baselines
├── sim_data/       # retained Chengdu source assets required by reproduction tools
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
- `repro/am_fidelity.py` — AM fidelity baseline.
- `repro/baseline_carpool.py` — PointerNet and common carpool baseline utilities.

The active `main` branch intentionally does **not** contain the later RoadMetric or hierarchical-decoder research extensions.

## Reproduction notes

See [docs/REPRO_AUDIT.md](docs/REPRO_AUDIT.md) for provenance, historical-code issues, and protocol details.

The old `code/` directory contained the three-year-old public implementation and vendored third-party baseline repositories. It is no longer used by the verified reproduction and has been removed from active branches; its complete Git history is archived separately.
