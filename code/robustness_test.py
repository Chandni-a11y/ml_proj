from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target


# Test multiple random splits
random_states = [42, 10, 20, 30, 40]

print("ROBUSTNESS TEST")
print("===============")


for random_state in random_states:

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y
    )

    # Create model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predictions and confidence
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    # Baseline accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Errors
    wrong = predictions != y_test
    total_errors = wrong.sum()

    # 90% confidence threshold
    threshold = 0.90

    accepted = confidence >= threshold
    flagged = confidence < threshold

    accepted_count = accepted.sum()
    flagged_count = flagged.sum()

    # Accuracy of accepted predictions
    if accepted_count > 0:
        accepted_accuracy = accuracy_score(
            y_test[accepted],
            predictions[accepted]
        )
    else:
        accepted_accuracy = 0

    # Errors flagged for review
    flagged_errors = wrong[flagged].sum()

    # Error rejection rate
    if total_errors > 0:
        error_rejection_rate = (
            flagged_errors / total_errors
        )
    else:
        error_rejection_rate = 0

    # Coverage
    coverage = accepted_count / len(y_test)

    print("\nRandom state:", random_state)
    print("Baseline accuracy:", round(accuracy, 4))
    print("Total errors:", total_errors)
    print("Accepted predictions:", accepted_count)
    print("Flagged for review:", flagged_count)
    print("Coverage:", round(coverage * 100, 2), "%")
    print(
        "Accepted prediction accuracy:",
        round(accepted_accuracy, 4)
    )
    print("Errors flagged:", flagged_errors)
    print(
        "Error rejection rate:",
        round(error_rejection_rate * 100, 2),
        "%"
    )