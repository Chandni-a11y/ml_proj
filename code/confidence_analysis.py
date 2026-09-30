from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target


# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 3. Create and train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)


# 4. Get predictions and confidence
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
confidence = probabilities.max(axis=1)


# 5. Check whether predictions are correct
correct = predictions == y_test


# 6. Define confidence ranges
ranges = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01)
]


print("Confidence vs Actual Accuracy")
print("-----------------------------")


# 7. Analyze each confidence range
for lower, upper in ranges:

    selected = (confidence >= lower) & (confidence < upper)

    count = selected.sum()

    if count > 0:
        actual_accuracy = correct[selected].mean()
    else:
        actual_accuracy = 0

    print("\nConfidence range:",
          int(lower * 100), "% -",
          int(min(upper, 1.0) * 100), "%")

    print("Number of predictions:", count)

    print(
        "Actual accuracy:",
        round(actual_accuracy * 100, 2),
        "%"
    )