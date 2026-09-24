# NetFlow v3 Results and Discussion

## Cross-host validation

The final validation used the NF-UNSW-NB15-v3 NetFlow dataset with direction-masked, forward-oriented metadata. Reverse-direction and explicitly bidirectional feature families were excluded before modeling. The retained information consisted of protocol indicators, packet-size aggregates, source-to-destination throughput, source-to-destination inter-arrival statistics, and flow timing fields. No packet payload was used.

The data contained 40 source hosts and four attack-bearing source hosts. The final fold design rotated both attack-bearing source hosts and benign source-host groups. Each fold therefore evaluated the model on traffic from source domains that were not used for training. Twelve folds were evaluated with three training epochs and a 300,000-row training subset per fold. Evaluation used all rows in each fold, avoiding the single-class sampling problem that occurs when only the first rows of an ordered file are evaluated.

The full host-held-out model achieved a mean AUROC of 0.9988 with a standard deviation of 0.0024. Mean average precision was 0.9994 with a standard deviation of 0.0011. Mean recall was 0.9910 with a standard deviation of 0.0302. These results show that the model preserves strong attack-ranking ability when both attack and benign source domains are held out.

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| AUROC | 0.9988 | 0.0024 | 0.9936 | 1.0000 |
| Average precision | 0.9994 | 0.0011 | 0.9964 | 1.0000 |
| Precision | 0.8932 | 0.1373 | 0.5736 | 1.0000 |
| Recall | 0.9910 | 0.0302 | 0.8950 | 1.0000 |
| F1 score | 0.9332 | 0.0828 | 0.7290 | 0.9981 |
| Matthews correlation coefficient | 0.7721 | 0.2945 | 0.1722 | 0.9978 |

## Threshold transfer and operational alerting

Ranking performance and operational threshold performance diverged. A single global threshold was selected from pooled calibration traffic at a nominal 1% benign false-positive target. The resulting threshold was applied unchanged to every unseen test host.

The full host-held-out global-threshold evaluation produced a mean benign false-positive rate of 19.84%, with a standard deviation of 32.26 percentage points. The mean false-alert rate was 4.619 alerts per hour, with a standard deviation of 6.071. The fold range was 0.023 to 20.983 false alerts per hour. Attack recall remained very high, with a mean of 99.986%.

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Benign false-positive rate | 19.84% | 32.26 percentage points | 0.007% | 91.63% |
| Attack recall | 99.986% | 0.022 percentage points | 99.941% | 100.000% |
| False alerts per hour | 4.619 | 6.071 | 0.023 | 20.983 |

The result demonstrates that high AUROC does not imply reliable threshold transfer. The model can rank attacks above benign flows while its score distribution changes substantially between benign source domains. A threshold that is appropriate for one host can therefore generate excessive alerts on another host.

## Adaptation-window limitation

A host-adaptive calibration experiment was considered using the earliest 20% of the unseen test host as an adaptation period. This protocol was designed to select the window by time alone and to prevent label-based filtering. The audit showed that the earliest window for fold 01 contained an attack rate of 56.77%. The later evaluation window contained an attack rate of 71.48%.

Consequently, the available dataset does not provide a valid benign-only adaptation period for this experiment. The adaptation rows cannot be used to estimate a benign score threshold without contaminating the calibration procedure with attack traffic. The host-adaptive result was therefore not treated as evidence and is excluded from the final performance claims.

A deployment-oriented adaptation study requires a separately identified benign-only observation period collected before the attack evaluation. Such a period is not available in the current experiment without using labels or introducing selection leakage.

## Research conclusion

The evidence supports the following conclusion:

> A causal neural network using forward-oriented NetFlow metadata generalizes strongly for attack ranking across unseen source hosts. However, unseen benign-host variation produces severe score-scale shift, so a single global threshold does not provide reliable operational alerting under the current calibration procedure.

The system should therefore be characterized as a high-quality research detector for attack prioritization and ranking rather than as a deployment-ready fixed-threshold alerting system. Further operational development requires benign-only host adaptation data, host-normalized features, quantile-based calibration, or domain-adaptive training. Any such method must be evaluated using the same full host-held-out protocol and must report benign false-positive rate, false alerts per hour, attack recall, and detection delay separately from AUROC.

## Reproducibility artifacts

The results in this section are derived from the following experiment artifacts:

- `reports/nfv3_full_cv/summary.csv`
- `reports/nfv3_full_cv/global_threshold_metrics.csv`
- `reports/nfv3_full_cv/global_threshold_metrics.json`
- `reports/nfv3_adaptation/fold_01/manifest.json`
- `reports/NFV3_FULL_HOST_THRESHOLD_REPORT.md`

## References

No external sources were required for this project-specific results section. All numerical results are derived from the local experiment artifacts listed above.
