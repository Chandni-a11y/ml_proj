# Teaching a Machine Learning System When Not to Trust Its Own Prediction

## Project Overview

This project investigates whether a machine learning system can identify
when its own prediction may be unreliable.

Machine learning models can sometimes produce confident predictions even
when the input data is different from the data used during training.
Therefore, accuracy alone may not be enough to determine whether a
prediction should be trusted.

The project investigates confidence, calibration, distribution shift,
and abstention as approaches for evaluating prediction reliability.

## Problem Statement

Can an ML system recognize when it should not trust its own prediction?

## Dataset

The project uses the Breast Cancer dataset available through
scikit-learn.

The dataset contains labelled classification data. It was divided into
training and testing sets using a 75:25 split with stratification.

Training samples: 426  
Testing samples: 143

## Baseline Model

A Logistic Regression classifier with feature standardization was used
as the baseline model.

The baseline achieved:

- Accuracy: 98.6%
- Test samples: 143
- Errors: 2

## Experiments

### 1. Confidence Analysis

Prediction probabilities were obtained from the model and the maximum
class probability was used as a confidence signal.

Average confidence:

- Correct predictions: 96.04%
- Wrong predictions: 75.49%

### 2. Calibration

Sigmoid probability calibration was tested.

Brier Scores:

- Original model: 0.0181
- Calibrated model: 0.0273

The original model had the lower Brier Score on this test split.

### 3. Distribution Shift

A controlled shift was introduced by increasing one test feature by 50%.

Results:

- Original accuracy: 98.6%
- Shifted accuracy: 95.1%
- Original average confidence: 95.75%
- Shifted average confidence: 94.96%

This showed that model performance decreased while average confidence
remained high.

### 4. Abstention

Different confidence thresholds were tested. Predictions below the
threshold were flagged for human review.

At a 90% threshold:

- Accepted predictions: 125
- Flagged for review: 18
- Coverage: 87.41%
- Accepted prediction accuracy: 100%
- Errors flagged: 2 out of 2
- Error rejection rate: 100%

These results apply only to the particular test split used in the
experiment.

### Robustness Testing

The reliability strategy was tested across five different train-test splits
using random states 42, 10, 20, 30, and 40.

Average results:
- Baseline accuracy: 98.60%
- Coverage at 90% confidence: 89.09%
- Accepted prediction accuracy: 99.53%
- Average error rejection rate: 68.75% across the four runs that contained errors

The results show that accepted predictions remained highly accurate across
the tested splits. However, the percentage of errors detected varied between
splits, showing that confidence-based abstention cannot identify every wrong
prediction.

### 5. Reliability Under Distribution Shift

At the 90% threshold:

Original data:

- Coverage: 87.41%
- Accepted prediction accuracy: 100%

Shifted data:

- Coverage: 83.22%
- Accepted prediction accuracy: 100%

The distribution shift used in this experiment was synthetic and should
not be considered a complete real-world OOD evaluation.

## Main Findings

The experiments provide initial evidence that confidence can provide
useful information about prediction reliability.

The abstention experiment showed that a model can avoid automatically
accepting some predictions and instead flag them for human review.

However, high confidence remained high under the controlled distribution
shift even though overall accuracy decreased.

Therefore, confidence alone may not be sufficient to identify all
unreliable or unfamiliar inputs.

## Limitations

- The test set contains only 143 samples.
- Only Logistic Regression was used as the baseline model.
- Only sigmoid calibration was tested.
- The distribution shift was synthetically created.
- The baseline produced only 2 errors on the test split.
- The current approach mainly uses prediction confidence rather than a
  complete uncertainty estimation method.

## Future Work

Future work can investigate:

- Additional uncertainty estimation methods
- Other probability calibration methods
- Out-of-distribution detection
- More realistic distribution shifts
- Larger and more diverse datasets
- Multiple classification models
- Multiple train-test splits or cross-validation
- More systematic confidence-threshold selection
- Combining confidence with OOD or uncertainty signals

## Project Structure

```text
ML_PROJECT/
│
├── code/
│   ├── baseline.py
│   ├── confidence_analysis.py
│   ├── reliability.py
│   └── shifted_reliability.py
│
├── documentation/
│   ├── conclusion.txt
│   ├── findings.txt
│   ├── limitations_future_work.txt
│   └── methodology.txt
│
├── results/
│   └── results_summary.py
│
└── README.md