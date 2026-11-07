# Evaluation: My harness for Guided SSIP / STEG — goal settling

This page describes the original Oscillate-only comparison. See the [expanded comparison](experiments/README.md) for Intercept, Oscillate, and Chase. Seeds 500–549 are repeated released evaluation; seeds 550–599 form a separate predeclared robustness extension. Do not pool the cohorts.

## Budget

250 physical actions, 1.0x. The original 200-action obstacle-placement reference is separately recorded benchmark preparation.

## Interpretation

- Primary success is final reward >0.85 and no collision, following the paper. Released code uses maximum reward instead.

- Under the secondary release criterion, released-seed counts are 26/50 versus 25/50; fresh counts are 26/50 versus 26/50.

- Earlier released-seed aggregate was known; the fresh extension is reported separately.

- Paper seed manifest is unverified and timing is on shared hardware. These results do not establish unseen robot task types.

## Reproduction discipline

Use the reported condition only. Establish native-path parity, prepare upstream models/assets and declare case manifests before expensive runs. This export itself is a source publication, not a new evaluation. Keep all historical and new cohorts separate.
