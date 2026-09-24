# Full Host-Held-Out NetFlow v3 Threshold Transfer

## Definitive finding

The full host-held-out experiment evaluated both attack-host and benign-host generalization. The model retained excellent ranking performance, but one global alert threshold did not transfer reliably across unseen benign hosts.

The full-host global-threshold evaluation produced a mean benign false-positive rate of **19.84%**, a standard deviation of **32.26 percentage points**, and a range from **0.007% to 91.63%**. The mean false-alert rate was **4.62 alerts per hour**, with a standard deviation of **6.07** and a range from **0.023 to 20.983 alerts per hour**. Attack recall remained extremely high at **99.99%** on average.

This result is more important than the earlier attack-host-only evaluation because the benign source-host groups also changed between folds. It shows that the detector’s ranking ability is strong, but its score scale is highly dependent on the host and traffic domain.

## Full-host global-threshold results

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Benign false-positive rate | 19.84% | 32.26 percentage points | 0.007% | 91.63% |
| Attack recall | 99.986% | 0.022 percentage points | 99.941% | 100.000% |
| False alerts per hour | 4.619 | 6.071 | 0.023 | 20.983 |

The fold-level results were:

| Fold | False alerts | Benign FPR | Attack recall | False alerts/hour |
|---|---:|---:|---:|---:|
| 01 | 5,605 | 34.706% | 99.974% | 8.641 |
| 02 | 144 | 0.033% | 99.993% | 0.222 |
| 03 | 3,459 | 0.521% | 100.000% | 5.333 |
| 04 | 3,691 | 0.813% | 100.000% | 5.690 |
| 05 | 80 | 0.037% | 99.993% | 0.123 |
| 06 | 846 | 75.468% | 100.000% | 1.304 |
| 07 | 15 | 0.007% | 99.941% | 0.023 |
| 08 | 3,602 | 91.631% | 100.000% | 5.553 |
| 09 | 124 | 2.439% | 100.000% | 0.191 |
| 10 | 55 | 0.026% | 99.941% | 0.085 |
| 11 | 4,724 | 29.251% | 99.986% | 7.283 |
| 12 | 13,611 | 3.094% | 100.000% | 20.983 |

## Comparison with attack-host-only validation

The earlier attack-host-only experiment kept the benign test groups largely fixed. Its pooled global-threshold result was substantially better, with approximately 1.63% pooled benign false-positive rate and 0.227 false alerts per hour. The full-host result is much worse because benign traffic now comes from different held-out source-host groups.

This comparison identifies benign-domain shift as a major source of threshold instability. The model still separates attack and benign traffic in ranking terms, but the absolute score distribution changes between benign environments.

## Interpretation

The model should currently be described as a **high-quality attack-ranking model**, not as a fixed-threshold production detector. AUROC and average precision remain very high, and recall is almost perfect. Yet operational alerting depends on score calibration, and calibration does not transfer across unseen benign hosts.

The most defensible conclusion is:

> A causal model using forward-oriented NetFlow metadata generalizes strongly in attack ranking, but unseen benign-host variation causes severe score-scale shift. A single global threshold is therefore insufficient for reliable operational alerting without additional domain calibration.

## Methodological validity

This experiment is stronger than the earlier fold design because each fold rotates both attack-bearing source hosts and benign source-host groups. The evaluation therefore tests two forms of generalization simultaneously. The threshold was selected from pooled calibration traffic and applied unchanged to all test hosts.

The source archive contains both-direction traffic, so the experiment remains direction-masked rather than a proof of physical one-way collection. Reverse-direction and bidirectional feature families were excluded before modeling.

## Recommended model-development direction

The next model-development stage should focus on score calibration and host normalization rather than simply increasing model size. Candidate methods include robust per-host normalization using a small benign-only adaptation window, quantile or rank-based score calibration, domain-adversarial training, and calibration models trained only on source-disjoint calibration hosts.

Any proposed method must be evaluated with the same full host-held-out protocol. The final report should retain false alerts per hour as the primary operational metric and AUROC as a separate ranking metric.

## Evidence files

- `reports/nfv3_full_cv/summary.csv`
- `reports/nfv3_full_cv/global_threshold_metrics.csv`
- `reports/nfv3_full_cv/global_threshold_metrics.json`
- `reports/nfv3_full_host_folds/fold_01/manifest.json`

## References

No external sources were required for this project-specific validation report. All numerical results are derived from the local experiment artifacts listed above.
