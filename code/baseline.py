from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 3. Create the ML model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])


# 4. Train the model
model.fit(X_train, y_train)


# 5. Make predictions
predictions = model.predict(X_test)

# 6. Get prediction probabilities
probabilities = model.predict_proba(X_test)

# 7. Get confidence for each prediction
confidence = probabilities.max(axis=1)

# 8. Measure accuracy
accuracy = accuracy_score(y_test, predictions)

print("Baseline Model Results")
print("----------------------")
print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))
print("Accuracy:", round(accuracy, 4))

print("\nFirst 10 Predictions with Confidence:")
print("--------------------------------------")

for i in range(10):
    print(
        "Prediction:", predictions[i],
        "| Confidence:", round(confidence[i] * 100, 2), "%",
        "| Actual:", y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i]
    )

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 9. Find high-confidence wrong predictions
wrong_predictions = predictions != y_test

print("\nHigh-Confidence Wrong Predictions:")
print("-----------------------------------")

found = False

for i in range(len(y_test)):
    if wrong_predictions[i] and confidence[i] >= 0.90:
        actual = y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i]

        print(
            "Prediction:", predictions[i],
            "| Confidence:", round(confidence[i] * 100, 2), "%",
            "| Actual:", actual
        )
        found = True

if not found:
    print("No wrong prediction with confidence >= 90%.")

# 10. Compare confidence of correct and wrong predictions
correct_predictions = predictions == y_test

correct_confidence = confidence[correct_predictions]
wrong_confidence = confidence[~correct_predictions]

print("\nConfidence Analysis:")
print("--------------------")

print(
    "Average confidence on correct predictions:",
    round(correct_confidence.mean() * 100, 2), "%"
)

print(
    "Average confidence on wrong predictions:",
    round(wrong_confidence.mean() * 100, 2), "%"
)

# 11. Create a calibrated version of the model
calibrated_model = CalibratedClassifierCV(
    model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train, y_train)

# 12. Get calibrated predictions and confidence
calibrated_predictions = calibrated_model.predict(X_test)
calibrated_probabilities = calibrated_model.predict_proba(X_test)
calibrated_confidence = calibrated_probabilities.max(axis=1)

# 13. Measure calibrated accuracy
calibrated_accuracy = accuracy_score(
    y_test,
    calibrated_predictions
)

print("\nCalibrated Model Results:")
print("------------------------")
print(
    "Calibrated Accuracy:",
    round(calibrated_accuracy, 4)
)

print(
    "Average calibrated confidence:",
    round(calibrated_confidence.mean() * 100, 2),
    "%"
)

print("\nFirst 10 Calibrated Predictions:")
print("---------------------------------")

for i in range(10):
    actual = y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i]

    print(
        "Prediction:", calibrated_predictions[i],
        "| Confidence:", round(calibrated_confidence[i] * 100, 2), "%",
        "| Actual:", actual
    )

# 14. Calibration evaluation
# Probability of class 1
original_probability = probabilities[:, 1]
calibrated_probability = calibrated_probabilities[:, 1]

# Brier Score
original_brier = brier_score_loss(y_test, original_probability)
calibrated_brier = brier_score_loss(y_test, calibrated_probability)

print("\nCalibration Evaluation:")
print("-----------------------")
print("Original Brier Score:", round(original_brier, 4))
print("Calibrated Brier Score:", round(calibrated_brier, 4))

# Calibration curve
original_fraction, original_mean = calibration_curve(
    y_test,
    original_probability,
    n_bins=5
)

calibrated_fraction, calibrated_mean = calibration_curve(
    y_test,
    calibrated_probability,
    n_bins=5
)

print("\nCalibration Curve:")
print("------------------")

print("Original model:")
for confidence_level, actual_accuracy in zip(
    original_mean,
    original_fraction
):
    print(
        "Predicted:", round(confidence_level * 100, 2),
        "% | Actual:", round(actual_accuracy * 100, 2), "%"
    )

print("\nCalibrated model:")
for confidence_level, actual_accuracy in zip(
    calibrated_mean,
    calibrated_fraction
):
    print(
        "Predicted:", round(confidence_level * 100, 2),
        "% | Actual:", round(actual_accuracy * 100, 2), "%"
    )

# 15. Create distribution-shifted test data
X_test_shifted = X_test.copy()

# Increase the first feature by 50%
X_test_shifted[:, 0] = X_test_shifted[:, 0] * 1.5

# Get predictions and confidence on shifted data
shifted_predictions = model.predict(X_test_shifted)
shifted_probabilities = model.predict_proba(X_test_shifted)
shifted_confidence = shifted_probabilities.max(axis=1)

# Calculate shifted accuracy
shifted_accuracy = accuracy_score(
    y_test,
    shifted_predictions
)

print("\nDistribution Shift Results:")
print("---------------------------")
print("Original Test Accuracy:", round(accuracy, 4))
print("Shifted Test Accuracy:", round(shifted_accuracy, 4))

print(
    "Average confidence on shifted data:",
    round(shifted_confidence.mean() * 100, 2),
    "%"
)

# Find high-confidence wrong predictions
shifted_wrong = shifted_predictions != y_test

high_confidence_wrong = (
    shifted_wrong & (shifted_confidence >= 0.90)
)

print(
    "High-confidence wrong predictions:",
    high_confidence_wrong.sum()
)