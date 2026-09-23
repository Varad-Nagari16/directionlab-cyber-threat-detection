# Direction-Preserving Data Contract

## Observation unit

One row represents one flow record observed from a source endpoint toward a destination endpoint during a defined observation interval. The row is not a reconstructed bidirectional session.

## Canonical fields

| Field | Type | Allowed use | Requirement |
|---|---|---|---|
| `record_id` | string | Audit only | Unique after ingestion; never a model feature |
| `timestamp` | UTC datetime | Model and split logic | Must be parseable and ordered |
| `duration_s` | float | Model feature | Non-negative; source-defined semantics recorded |
| `bytes` | float | Model feature | Non-negative |
| `packets` | float | Model feature | Non-negative |
| `bytes_per_s` | float | Model feature | Derived only from current row and duration policy |
| `packets_per_s` | float | Model feature | Derived only from current row and duration policy |
| `transport_protocol` | categorical | Model feature | Source-defined mapping documented |
| `src_port` | integer/categorical | Model feature | Retain observed source orientation |
| `dst_port` | integer/categorical | Model feature | Retain observed destination orientation |
| `protocol_flags` | categorical/vector | Model feature | Only flags visible in the observed record |
| `label` | categorical | Target only | Mapping versioned and documented |
| `dataset` | categorical | Audit and evaluation | Not a model feature in cross-dataset tests |
| `host_group` | string | Split control only | Excluded from model inputs |
| `split` | categorical | Pipeline control only | Assigned after split construction |

## Prohibited inputs

The following fields must not be used by the primary detector:

- reverse-flow byte, packet, duration, rate, or flag fields;
- bidirectional totals or session aggregates formed by joining opposite directions;
- payload bytes, payload contents, decrypted application data, or content signatures;
- future records relative to the prediction timestamp;
- test-set statistics used to fit encoders, scalers, imputers, thresholds, or calibration models;
- source or destination identifiers when they can act as host memorisation rather than traffic evidence;
- labels, attack annotations, or post-event fields;
- random row identifiers or file-order artefacts.

## Causal-history rule

For a target record at time `t`, history may contain only records available at or before `t`, according to the declared observation scope. The implementation must define whether history is global, per source host, per monitored direction, or per collection stream. If records have equal timestamps, use a stable source ordering and document it. No future row may enter a temporal window because of sorting, padding, batching, or data-loader behaviour.

## Required validation checks

The ingestion and feature pipeline must test non-negative numeric constraints, timestamp parsing, duplicate identifiers, label coverage, missingness, invalid port ranges, protocol mapping, stable column types, and the absence of forbidden columns. The temporal pipeline must test that every history index is less than or equal to the target index and that the receptive field is causal.

## Versioning

Every processed dataset and model checkpoint must record:

- source dataset name, version, and file checksum;
- schema version;
- label-map version;
- feature-contract version;
- split configuration;
- preprocessing artifact checksum;
- model and code version;
- random seed;
- training timestamp.

## Alert record minimum

```json
{
  "timestamp": "2026-09-19T10:42:07Z",
  "flow_id": "directional-record-id",
  "threat_class": "volumetric_attack",
  "confidence": 0.91,
  "severity": "HIGH",
  "evidence": {
    "bytes": 8460000,
    "packets_per_s": 1200,
    "observed_direction": "source_to_destination"
  },
  "model_version": "unassigned",
  "feature_contract_version": "0.1.0"
}
```

The example is a schema illustration only. It is not an experimental result or a production severity policy.
