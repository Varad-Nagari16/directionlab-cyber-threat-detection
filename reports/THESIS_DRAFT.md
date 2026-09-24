# AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

## Abstract

This project presents DirectionLab, a causal machine-learning framework for detecting cyber threats from network-flow metadata without inspecting payloads or using explicit reverse-flow features. The framework combines a static multilayer perceptron, a causal temporal convolutional network, and a gated fusion layer. The model processes the current flow together with a bounded history of earlier observations. It does not access future observations or a reverse-flow feature pathway.

The study uses the CIC-IDS2017 MachineLearningCVE flow files as a reproducible prototype dataset. A conservative feature contract retains 26 forward-oriented metadata fields and removes identifiers, payload-related fields, unrecognized fields, and backward-flow feature columns. A chronological binary experiment trains on Monday through Wednesday, calibrates on Thursday, and tests on the unseen Friday period. The model obtains an AUROC of 0.805, precision of 92.77%, recall of 28.03%, F1 of 43.06%, and Matthews correlation coefficient of 0.396 on the complete Friday test set. A separate stratified family-recognition experiment obtains a tuned macro-F1 of 86.22%. DDoS and PortScan recognition is strong in the stratified experiment, while Bot precision remains limited at 23.89% after calibration.

The results show that causal flow metadata can provide useful threat ranking and alerting capability, but performance changes substantially across traffic days and attack families. Because CIC-IDS2017 does not establish that its flow aggregates were collected at a physically one-way observation point, the study presents the result as a direction-preserving flow-based prototype rather than a complete proof of deployment on genuine unidirectional traffic.

**Keywords:** unidirectional traffic, flow metadata, causal temporal model, network intrusion detection, anomaly detection, CIC-IDS2017, IPFIX.

## 1. Introduction

Network monitoring systems may be unable to inspect payloads because traffic is encrypted, privacy controls restrict content collection, or the monitoring point observes only one direction of a communication. These conditions motivate detectors that use metadata such as duration, packet counts, byte counts, rates, ports, and transport flags.

A detector for this setting must satisfy stronger constraints than an ordinary bidirectional flow classifier. It must avoid reverse-flow leakage, must not use future records, and must report evidence that can be interpreted by an operator. DirectionLab addresses these requirements at the model-input and temporal-architecture levels.

The project asks the following research question:

> Can a causal model using direction-preserving, payload-independent flow metadata provide useful cyber-threat detection without explicit reverse-flow access?

The work makes three contributions. First, it defines an allowlisted feature contract that excludes explicit reverse-flow and payload features. Second, it implements a causal temporal detector with static and historical encoders. Third, it separates temporal generalization from attack-family recognition so that high random-split scores are not mistaken for future-environment performance.

## 2. Background and Research Gap

Intrusion detection systems identify activity that may violate security policy or indicate compromise. NIST describes intrusion detection and prevention systems as technologies that monitor events and analyze them for signs of possible incidents [1]. Flow-based detection is attractive because flow records are smaller and less sensitive than packet payloads, and IPFIX provides a standardized protocol for exporting traffic-flow information [2].

CIC-IDS2017 provides labeled network-flow records and attack scenarios for intrusion-detection evaluation [3]. It is useful for reproducible prototyping, but its flow-level aggregates do not by themselves establish that a record was created from a physically one-way monitoring point. This distinction is central to the present work.

Many evaluation procedures also use random row splits. Such splits are convenient for measuring class recognition, but related observations from the same capture period may appear in both training and testing. A model can therefore achieve a high score without demonstrating robustness to a future day, unseen host, or changed environment.

The research gap addressed here is the lack of a unified prototype that combines a strict feature allowlist, causal history, chronological evaluation, threshold calibration, and evidence-bearing alerts while explicitly separating what the dataset proves from what it cannot prove.

## 3. Proposed Method

### 3.1 Observation contract

The observation unit is one prepared flow record. The retained features include destination port, flow duration, forward packet counts, forward byte and packet-length statistics, flow rates, forward inter-arrival statistics, selected transport flags, and average packet size. The model excludes source and destination identifiers, payload fields, backward-flow fields, and unrecognized columns.

The current CIC-IDS2017 files do not provide a reliable raw timestamp and protocol field in the extracted CSVs used here. The preparation pipeline preserves capture-file and row order as a temporal proxy. This choice supports reproducible chronology but does not replace packet-level temporal provenance.

### 3.2 Causal architecture

The model has four stages. A static encoder maps the current feature vector into a hidden representation. A causal temporal convolutional network maps the previous 32 observations and the current observation into a historical representation. The temporal convolutions are left-padded, so the output at time *t* cannot depend on records after *t*. A learned gate controls how much temporal information enters the fused representation.

The fused representation feeds a binary head, a multiclass head, and a severity head. The binary head predicts benign versus attack. The multiclass head predicts BENIGN, DDoS, PortScan, Bot, or OTHER_ATTACK. JSON alerts include the predicted class, confidence scores, threshold, severity, and offline reference label when a label is available.

### 3.3 Directionality audit

The saved checkpoint is audited for forbidden feature names. The audit checks for explicit backward-flow terms, payload terms, and endpoint identifiers. A passing audit demonstrates that the checkpoint feature list contains no explicit forbidden fields. It cannot prove that the original dataset was collected from a one-way sensor. Collection-level provenance remains necessary for that claim.

## 4. Experimental Design

### 4.1 Temporal binary experiment

The primary experiment uses a chronological split. Monday, Tuesday, and Wednesday provide training data. Thursday provides calibration data. Friday is held out as an unseen test period. The model is trained with a binary objective. The decision threshold is calibrated on Thursday at a target false-positive rate of 1% and is then applied without retuning to Friday.

This design tests temporal distribution shift more realistically than a random row split. It also exposes the difference between ranking quality and operational alert quality. AUROC measures ranking across thresholds, while precision, recall, F1, MCC, and false-positive rate depend on the selected operating point.

### 4.2 Family-balanced multiclass experiment

The secondary experiment creates an 80/10/10 stratified split within each family. Every family therefore appears in training, calibration, and testing. This experiment measures class recognition rather than unseen-day generalization. Remaining non-benign labels are grouped into OTHER_ATTACK.

The Bot class receives a calibrated rejection threshold selected from calibration alerts. If a prediction is Bot but its Bot score is below the selected threshold, the system chooses the strongest non-Bot alternative. The test set is used only once for final reporting.

## 5. Results

### 5.1 Temporal binary results

| Measure | Friday result |
|---|---:|
| Rows evaluated | 703,245 |
| AUROC | 0.805 |
| Average precision | 0.747 |
| Precision | 92.77% |
| Recall | 28.03% |
| F1 | 43.06% |
| MCC | 0.396 |
| Alert rate | 12.42% |

The confusion counts are 80,996 true positives, 408,009 true negatives, 6,313 false positives, and 207,927 false negatives. The result indicates a high-precision but low-recall operating point. The detector produces relatively few incorrect alerts, but it misses a large fraction of attacks under the selected threshold.

The binary result is strongly family dependent. DDoS recall is approximately 63.15%. PortScan recall is approximately 0.09%, and Bot recall is 0%. Therefore, the binary model should be presented as a general attack-ranking prototype rather than a uniformly effective detector for every attack family.

### 5.2 Stratified multiclass results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| BENIGN | 99.95% | 98.98% | 99.46% |
| DDoS | 99.85% | 99.94% | 99.89% |
| PortScan | 99.29% | 99.89% | 99.59% |
| Bot | 23.89% | 68.53% | 35.43% |
| OTHER_ATTACK | 93.80% | 99.82% | 96.72% |

The tuned multiclass experiment obtains macro-F1 of 86.22% and weighted-F1 of 99.18%. DDoS and PortScan are recognized with high precision and recall. Bot recall remains useful but Bot precision is low. The Bot test support is only 197 rows, so the Bot estimate is less stable than estimates for the larger classes.

### 5.3 Alert output

The system writes JSON Lines alerts. A typical record contains the predicted class, class scores, aggregate attack score, severity, source file, and offline reference label. The reference label is used for evaluation only and would not be available in real deployment.

## 6. Discussion

The temporal result supports the hypothesis in a limited form. Causal flow metadata provides useful information for distinguishing threats from benign activity, even when the model is restricted to a forward-oriented feature contract. However, the low recall on the unseen Friday period shows that the learned decision boundary is sensitive to traffic distribution and attack family.

The multiclass result demonstrates that the architecture can recognize attack families when examples of those families are present in training. Its high random-split score must not be compared directly with the temporal binary score. The experiments answer different questions and use different split policies.

The Bot result shows why macro-F1 and per-class reporting are necessary. Weighted-F1 is dominated by the largest classes and can conceal a weak minority class. Bot threshold calibration improves Bot F1 from 27.35% to 35.43%, but the resulting precision of 23.89% remains insufficient for an unqualified operational claim.

## 7. Limitations

The most important limitation is dataset provenance. CIC-IDS2017 flow aggregates may summarize both directions of a connection. The feature allowlist removes explicit reverse-flow columns, but it cannot retroactively prove that reverse packets were unavailable during original feature extraction.

The temporal proxy also uses capture order because the prepared CSVs omit a reliable raw timestamp and protocol field. This preserves an ordered evaluation procedure but is weaker than packet-level time provenance.

The Bot class is small. Additional Bot examples are necessary before drawing strong conclusions. The study also does not yet report false alerts per hour, detection delay, host-held-out generalization, adversarial robustness, or calibration on an external dataset.

## 8. Conclusion and Future Work

DirectionLab provides a complete research prototype for causal threat detection from flow metadata. It implements an explicit feature contract, a causal temporal encoder, gated fusion, binary and multiclass heads, threshold calibration, JSON alerts, and reproducible evaluation.

The primary unseen-day experiment achieves AUROC 0.805 and precision 92.77% at recall 28.03%. The family-balanced experiment achieves tuned macro-F1 86.22%, with strong DDoS and PortScan recognition and limited Bot precision. These results support the feasibility of the approach while showing that generalization and attack-family coverage remain difficult.

The next validation stage requires genuinely direction-preserving data. Suitable data include one-way NetFlow or IPFIX, Zeek logs collected at a direction-limited monitoring point, or packet captures in which reverse traffic is removed before feature extraction. The future experiment should use host- and scenario-held-out splits, report false alerts per hour and detection delay, and preserve the separation between temporal generalization and family recognition.

## References

[1]: https://csrc.nist.gov/pubs/sp/800/94/final "NIST Guide to Intrusion Detection and Prevention Systems"
[2]: https://www.rfc-editor.org/info/rfc7011 "RFC 7011: Specification of the IP Flow Information Export Protocol"
[3]: https://www.unb.ca/cic/datasets/ids-2017.html "Canadian Institute for Cybersecurity: CIC-IDS2017 Dataset"
