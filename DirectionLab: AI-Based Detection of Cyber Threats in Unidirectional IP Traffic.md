DirectionLab-AI-Based-Detection-of-Cyber-Threats-in-Unidirectional-IP-Traffic.md

This repository contains the first implementation milestone for a direction-preserving, payload-independent, temporally causal network-threat detector.

## Current status

The initial model stack is implemented and verified:

- canonical schema validation fails closed on reverse-flow, bidirectional, payload, and future-looking fields;
- current-flow features include duration, bytes, packets, rates, directional ports, and transport indicators;
- causal windows are left-padded and contain only observations at or before the target row;
- the model combines a static multilayer perceptron with a dilated causal temporal convolutional network;
- gated fusion produces binary and multiclass outputs;
- four contract, feature, causality, and model-shape tests pass;
- the smoke-training loss decreases on a deterministic fixture.

The smoke fixture is an architecture check only. It is not a dataset result and must not be reported as a research experiment.

## Run the checks

From this directory:

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m directionlab.smoke_train
```

Expected output is four passing tests and a smoke-training status containing `"status": "ok"`.

## Repository map

```text
src/directionlab/data_contract.py  # schema and forbidden-feature validation
src/directionlab/features.py       # current-row features and causal windows
src/directionlab/models.py         # causal TCN and gated fusion model
src/directionlab/smoke_train.py    # deterministic architecture smoke test
tests/test_directionlab.py         # unit and leakage-oriented checks
configs/smoke.yaml                 # initial experiment contract
docs/PROJECT_PLAN.md               # end-to-end research plan
docs/DATA_CONTRACT.md              # canonical field policy
```

## Next real-data step

Select or upload the primary dataset before training a scientific model. The next implementation will add a source adapter, label map, chronological split builder, train-only preprocessing, baseline classifiers, calibration, and an evaluation report. UGR'16 is the recommended starting point for the research question, while CIC-IDS2017 can serve as an external validation source after its flow semantics are documented.

Until a real dataset is selected and inspected, all measured metrics remain **TBD**.
