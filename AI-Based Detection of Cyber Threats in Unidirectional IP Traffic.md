# AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

## 1. Project purpose

This project will develop and evaluate a machine-learning framework for detecting cyber threats when the monitoring system observes traffic in only one IP direction at a time. The detector will use flow metadata such as timestamps, duration, bytes, packets, rates, ports, protocol flags, and temporal history. It will not inspect payload contents, use future records, or derive features from the reverse flow.

The project has two outputs. The first is a reproducible research implementation with controlled experiments. The second is an evidence-bearing alert pipeline that can eventually feed a dashboard or security-monitoring workflow. No performance claim will be made until the data, split policy, leakage tests, calibration, and baselines have been completed.

## 2. Research questions

1. Can a direction-preserving detector identify malicious or anomalous unidirectional flows using metadata alone?
2. Does causal temporal context improve detection compared with a current-flow-only model?
3. How much performance is lost when hosts, time periods, datasets, or environments are unseen during testing?
4. Can the system produce calibrated probabilities and operational alerts without creating an unacceptable false-alert burden?

## 3. Explicit boundaries

The first version will be a research detector rather than a production intrusion-prevention system. It will make binary and, where labels permit, multiclass predictions. It will not decrypt traffic, inspect payloads, reconstruct sessions from reverse traffic, use future observations, or claim generalisation to operational networks without external validation.

A feature is valid only if it can be computed from the current observed direction and records available at or before the prediction time. Any feature that requires the opposite direction, a bidirectional conversation key, a future label, or information from the complete dataset will be rejected or isolated as an explicitly invalid comparison.

## 4. System architecture

```text
Raw flow records
      |
      v
Direction-preserving ingestion and schema validation
      |
      v
Canonical one-way flow table + data-quality report
      |
      +--> Current-flow features: volume, timing, ports, protocol flags
      |
      +--> Causal history: previous records within an allowed observation scope
      |
      v
Chronological and host-controlled split construction
      |
      +--> Baselines: rules, logistic regression, random forest, LightGBM
      |
      +--> Static encoder: MLP
      |
      +--> Causal encoder: dilated temporal convolutional network
      |
      v
Gated fusion and calibrated binary/multiclass heads
      |
      v
Metrics, uncertainty, evidence-bearing JSON alerts, dashboard-ready output
```

The first implementation should use a static baseline and a causal temporal model before adding a dashboard. This ordering makes it possible to distinguish data-pipeline problems from model improvements.

## 5. Development phases

### Phase 0 — Scope and reproducibility contract

Freeze the research questions, threat taxonomy, data sources, allowed feature families, split rules, random seeds, and success criteria. Create a machine-readable experiment configuration so that every reported number can be regenerated.

Deliverables: project charter, data contract, feature allowlist, label policy, experiment configuration, and an evidence log.

### Phase 1 — Dataset acquisition and inspection

Start with one primary labelled flow dataset and one external validation dataset. UGR'16 is a suitable candidate for flow-level evaluation because it was created for network intrusion-detection evaluation and contains long-duration network-flow traces [1]. CIC-IDS2017 is a suitable external benchmark because the Canadian Institute for Cybersecurity provides labelled benign and attack traffic for intrusion-detection research [2]. The exact files and licensing conditions must be recorded before inclusion.

Inspect file schemas, timestamp resolution, missing values, duplicate records, class distribution, source and destination fields, label quality, and whether the available records can be represented as one-way observations without reverse-flow features. Do not merge datasets until their semantics and label definitions have been documented.

Deliverables: raw-data manifest, schema report, label mapping, class-distribution report, and a reproducible ingestion script.

### Phase 2 — Direction-preserving data contract

Convert every source into a canonical table. Each row represents one observed direction of one flow record. The canonical table should contain a stable record identifier, observation timestamp, source and destination identifiers where permitted, transport protocol, source and destination ports, duration, bytes, packets, derived rates, protocol flags, label, dataset name, and split metadata.

The ingestion layer must preserve the observed source-to-destination orientation. It may use an explicitly provided flow direction from the source dataset, but it must not join a record to its reverse companion to manufacture features. Any identifier used only to control leakage must be excluded from model inputs.

Deliverables: `data_contract.md`, schema validation tests, rejected-row report, and a directionality audit.

### Phase 3 — Baseline feature pipeline

Implement deterministic feature transformations. The initial feature families should be volume and rate features, temporal features, transport and port metadata, and protocol flags. Fit all learned preprocessing steps on the training partition only. Save the feature names, dtypes, missing-value policy, scaling parameters, and categorical encodings with each experiment.

Build three baselines: a simple rule or majority-class reference, a linear probabilistic classifier, and a tree-based classifier. These baselines are necessary for demonstrating whether the temporal model adds value.

Deliverables: feature builder, baseline training script, baseline evaluation report, and feature-importance report.

### Phase 4 — Causal temporal model

Represent each prediction as the current flow plus a bounded history of earlier observations that are available under the declared observation scope. Start with a history length of 32 records or a 60-second window, but treat these values as configuration rather than scientific facts.

Implement a causal temporal convolutional network with dilations such as 1, 2, 4, and 8. The temporal encoder must use padding and indexing that prevent future records from entering the receptive field. Add a static encoder for current-flow features, then combine the static and temporal representations with a gated fusion layer.

Train separate binary and multiclass heads where the dataset supports a defensible attack-family mapping. Keep the first model small enough to run repeatably on CPU, then scale only after correctness has been established.

Deliverables: model module, causal-padding tests, training logs, checkpoint files, ablation configuration, and model card.

### Phase 5 — Evaluation and leakage testing

Use chronological train, validation, calibration, and test partitions. Add host-controlled or environment-controlled variants where the required identifiers are available. The primary test must remain untouched until model selection and calibration are complete.

Report average precision, area under the receiver operating characteristic curve, macro-F1, Matthews correlation coefficient, recall at fixed false-positive rates, false alerts per hour, detection delay, and calibration or Brier score. Report confidence intervals where repeated resampling or repeated splits are appropriate.

Run explicit negative tests: shuffled labels, reverse-flow feature injection in a forbidden branch, future-history injection, duplicate leakage, host overlap, and random-split comparison. A detector that improves only under a random split should not be presented as robust.

Deliverables: evaluation harness, leakage audit, ablation report, calibration report, and final test report.

### Phase 6 — Alert schema and operational prototype

Convert predictions into a stable JSON alert schema. Each alert should include an event identifier, observation timestamp, flow identifier, threat class, confidence, severity, evidence fields, model version, feature-contract version, and an explanation reference. The explanation must describe evidence available from the observed direction; it must not expose prohibited reverse-flow information.

Add a small dashboard only after the offline pipeline is reliable. The dashboard should show alert counts, confidence distribution, severity, false-positive review status, latency, and the current model/data-contract versions. It should be treated as a presentation layer, not as evidence of detection quality.

Deliverables: JSON schema, alert serializer, example alerts, dashboard prototype, and dashboard test data generated from real held-out records rather than random examples.

### Phase 7 — Reproducibility and research reporting

Package the experiment commands, environment specification, dataset manifest, configuration files, preprocessing artifacts, model checkpoints, metric outputs, and plots. Maintain a distinction between target thresholds and measured values. The poster's values marked as “TBD” must remain TBD until experiments produce them.

Write the final report in this order: problem definition, threat and observation model, data, feature contract, models, split and leakage controls, metrics, results, limitations, and deployment considerations. Include an appendix containing the exact configuration and commands used to produce each table and figure.

## 6. Initial repository layout

```text
cyber-threat-detection/
├── README.md
├── configs/                 # versioned experiment YAML files
├── data/
│   ├── raw/                 # never modified source files
│   ├── interim/             # normalized but not final data
│   └── processed/           # split-specific model inputs
├── docs/
│   ├── PROJECT_PLAN.md
│   └── DATA_CONTRACT.md
├── models/                  # checkpoints and model cards
├── notebooks/               # exploratory work only
├── reports/                 # generated tables, plots, and reports
├── src/
│   ├── ingest/              # source adapters and validation
│   ├── features/            # allowlisted feature transformations
│   ├── splits/              # chronological and leakage-controlled splits
│   ├── models/              # baselines and causal neural models
│   ├── evaluation/          # metrics, calibration, and audits
│   └── alerts/              # schema and serialization
└── tests/                   # unit, integration, and leakage tests
```

## 7. First milestone: a trustworthy data slice

The first coding milestone is not model training. It is a small, inspectable, direction-preserving data slice that passes validation.

The milestone is complete when:

1. One source dataset is documented and available under its permitted terms.
2. A loader converts it into the canonical one-way schema.
3. Validation reports missingness, duplicates, timestamp order, label counts, and invalid values.
4. A feature allowlist produces a model matrix without payload or reverse-flow fields.
5. A chronological split is created without fitting transforms on validation or test data.
6. Automated tests fail if a forbidden feature or future record enters the input.
7. A short data report can be regenerated from one command.

Only after this milestone should we implement the first baseline model.

## 8. Recommended default technical stack

Use Python 3.11 with pandas or Polars for tabular processing, scikit-learn for preprocessing and classical baselines, PyTorch for the causal temporal model, PyYAML for experiment configuration, pytest for tests, and Matplotlib or Seaborn for figures. Use Parquet for normalized data and JSON Schema for alert validation. Keep notebooks for inspection and use scripts or modules for every reproducible result.

## 9. Decisions that must be recorded before experiments

The project must record the selected primary dataset, exact source files and versions, label mapping, attack-family policy, observation scope, treatment of source and destination identifiers, handling of encrypted or missing transport metadata, history construction rule, split policy, calibration method, threshold-selection rule, and hardware/software versions.

These decisions are part of the scientific method. Changing them after seeing test results would make comparisons difficult and may introduce optimistic bias.

## References

[1]: https://nesg.ugr.es/nesg-ugr16/dataset_AuthorVersionFinal.pdf "UGR'16: A New Dataset for the Evaluation of Cyclostationarity-Based Network IDSs"

[2]: https://www.unb.ca/cic/datasets/ids-2017.html "Intrusion Detection Evaluation Dataset (CIC-IDS2017)"

[3]: https://docs.zeek.org/en/current/ "Zeek Documentation"
