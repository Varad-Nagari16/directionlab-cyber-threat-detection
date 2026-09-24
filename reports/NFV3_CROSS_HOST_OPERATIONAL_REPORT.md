# NetFlow v3 Cross-Host Validation and Operational Findings

## Overall conclusion

The direction-masked detector transfers strongly in **ranking quality** across unseen source hosts, but it does not yet provide a stable operational alerting threshold. Across 12 host-held-out folds, the model achieved a mean AUROC of **0.9957** with a standard deviation of **0.0064**. Mean recall was **0.9988** with a standard deviation of **0.0026**.

The operational evaluation changes the deployment conclusion. Using the threshold calibrated separately for each fold, the pooled benign false-positive rate was **56.78%**. The mean false-alert rate was **7.89 alerts per hour**, with a standard deviation of **5.45** and a range from **0.097 to 12.145 alerts per hour**. Thus, the current model is a strong research detector for ranking traffic, but it is **not deployment-ready as a fixed alerting system**.

## Experimental design

The experiment used the NF-UNSW-NB15 NetFlow v3 dataset after removing reverse-direction and bidirectional feature families. The retained inputs were forward-oriented metadata, including protocol indicators, forward packet-size aggregates, source-to-destination throughput, source-to-destination inter-arrival statistics, and flow timing fields. Payload content was not used.

The prepared data contained 40 source hosts, of which four contained attack traffic. The cross-validation plan generated 12 ordered combinations. Each combination used two attack-bearing hosts for training, one different attack-bearing host for calibration, and a fourth attack-bearing host as the unseen test host. Benign source hosts were kept disjoint across the train, calibration, and test partitions.

Thresholds were selected on the calibration host at a nominal target false-positive rate of 1%. The selected threshold was then applied to the corresponding unseen test host without test-time retuning.

## Ranking performance

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| AUROC | 0.9957 | 0.0064 | 0.9823 | 0.9998 |
| Average precision | 0.9970 | 0.0051 | 0.9854 | 1.0000 |
| Recall | 0.9988 | 0.0026 | 0.9928 | 1.0000 |
| Precision | 0.8662 | 0.0927 | 0.7772 | 0.9980 |
| F1 score | 0.9254 | 0.0512 | 0.8746 | 0.9984 |
| Matthews correlation coefficient | 0.5640 | 0.3024 | 0.3200 | 0.9929 |

The high AUROC and average precision show that the model generally assigns higher scores to attack flows than to benign flows, even when the attack-bearing source host is not present during training. The lower and more variable MCC indicates that ranking quality does not automatically produce a reliable binary operating point.

## Operational false-alert performance

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Benign false-positive rate | 0.5678 | 0.3919 | 0.0070 | 0.8738 |
| Attack recall | 0.9988 | 0.0026 | 0.9928 | 1.0000 |
| False alerts per hour | 7.8920 | 5.4473 | 0.0971 | 12.1452 |

Across all folds, the operational evaluator counted 61,430 false alerts among 108,192 benign test flows. This produces the pooled benign false-positive rate of 0.5678. The observed test windows were approximately 648.65 hours per fold. False alerts per hour were computed as false alerts divided by the observed `FLOW_START_MILLISECONDS` to `FLOW_END_MILLISECONDS` span.

The results are highly heterogeneous. Folds 08 and 09 produced low false-alert rates of 0.335 and 0.097 alerts per hour, respectively. Folds 01, 02, 03, 05, 07, 10, and 12 produced approximately 11.6 to 12.1 alerts per hour. This variation is consistent with host-dependent score-distribution shift.

## Threshold instability

The calibrated threshold varied from approximately **0.00008** to **0.99383** across folds. Such a large range means the model scores do not have a stable cross-host scale. A threshold that is conservative for one host can become excessively permissive or excessively strict for another host.

This finding is central to the research question. The model can recognize attack-like forward metadata patterns, but the current training and calibration procedure does not make the scores comparable enough for a single operational threshold to transfer reliably between hosts.

## Validity and limitations

The experiment is direction-masked rather than a proof of physical one-way collection because the source archive contains both-direction traffic. Reverse-direction feature families were excluded before modeling, but the original capture process was not physically restricted to one direction.

The test partitions are attack-heavy, with attack prevalence between approximately 75% and 80%. Precision and F1 therefore should not be interpreted as production alert quality. The benign false-positive rate and false alerts per hour are more informative for deployment assessment.

The false-alerts-per-hour estimate uses the observed flow timestamp span. It is an empirical rate over the test window, not a claim about the alert rate of a live sensor with a different flow volume or time distribution.

## Research conclusion

The evidence supports the following claim:

> A causal neural detector using forward-oriented NetFlow metadata can rank attack flows highly across unseen source hosts, but host-dependent score shift prevents reliable threshold transfer under the current calibration procedure.

The evidence does not support the stronger claim that the detector already maintains a 1% false-positive operating point on unseen hosts. Before deployment-oriented claims are made, the project should evaluate pooled calibration, fixed global threshold transfer, probability calibration, host-normalized features, and realistic benign-dominant traffic prevalence.

## Recommended next experiment

The next experiment should select one global threshold using only pooled calibration traffic from the training procedure and apply that unchanged threshold to every unseen test host. The resulting per-fold benign false-positive rate, false alerts per hour, recall, and detection delay should be reported. This will directly test whether a threshold can transfer across hosts without per-fold adaptation.

## Evidence files

The reported values were calculated from the project’s 12-fold cross-validation summary and operational metrics files:

- `reports/nfv3_cv/cross_validation_summary.csv`
- `reports/nfv3_cv/operational_metrics.csv`
- `data/processed/nfv3_host_folds/fold_01/manifest.json`

## References

No external sources were required for this project-specific results summary. The numerical findings are derived from the local experiment artifacts listed above.
