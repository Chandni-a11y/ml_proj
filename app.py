import streamlit as st
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="ML Reliability Demo",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Load dataset and train model
# --------------------------------------------------

@st.cache_resource
def train_model():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    model.fit(X_train, y_train)

    return model, data


model, data = train_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🤖 ML Reliability Demo")

st.subheader(
    "Teaching a Machine Learning System When Not to Trust Its Own Prediction"
)

st.write(
    "A demonstration of prediction confidence and "
    "confidence-based abstention using Logistic Regression."
)

st.info(
    "Educational demonstration using the Breast Cancer dataset. "
    "This application is not a clinical diagnostic tool."
)


# --------------------------------------------------
# Fixed reliability threshold
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.90


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.divider()

st.header("🔬 Enter Feature Values")

st.write(
    "The model uses 30 input features to generate a prediction "
    "and estimate its confidence."
)


default_values = np.mean(data.data, axis=0)

input_values = [None] * 30


# --------------------------------------------------
# Mean features
# --------------------------------------------------

st.subheader("Mean Features")

columns = st.columns(3)

for i in range(10):

    with columns[i % 3]:

        input_values[i] = st.number_input(
            data.feature_names[i],
            value=float(default_values[i]),
            format="%.4f"
        )


# --------------------------------------------------
# Error features
# --------------------------------------------------

st.subheader("Error Features")

columns = st.columns(3)

for i in range(10, 20):

    with columns[(i - 10) % 3]:

        input_values[i] = st.number_input(
            data.feature_names[i],
            value=float(default_values[i]),
            format="%.4f"
        )


# --------------------------------------------------
# Worst features
# --------------------------------------------------

st.subheader("Worst Features")

columns = st.columns(3)

for i in range(20, 30):

    with columns[(i - 20) % 3]:

        input_values[i] = st.number_input(
            data.feature_names[i],
            value=float(default_values[i]),
            format="%.4f"
        )


# --------------------------------------------------
# Prediction button
# --------------------------------------------------

st.divider()

predict = st.button(
    "🔍 Make Prediction",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Prediction result
# --------------------------------------------------

if predict:

    input_array = np.array(input_values).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    probabilities = model.predict_proba(input_array)[0]

    confidence = float(np.max(probabilities))

    predicted_class = data.target_names[prediction]


    st.divider()

    st.header("📋 Prediction Result")


    # Result summary
    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction",
            predicted_class.upper()
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    st.write(
        f"**Reliability Threshold:** "
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
    )


    # --------------------------------------------------
    # Reliability decision
    # --------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        st.success(
            "### ✅ Prediction Accepted"
        )

        st.write(
            "The model confidence is above the 90% reliability "
            "threshold."
        )

    else:

        st.warning(
            "### ⚠️ Flagged for Human Review"
        )

        st.write(
            "Confidence is below the 90% reliability threshold. "
            "The prediction has been flagged for human review."
        )


    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    st.subheader("Confidence Level")

    st.progress(confidence)

    st.caption(
        f"Model confidence: {confidence * 100:.2f}%"
    )


    # --------------------------------------------------
    # Class probabilities
    # --------------------------------------------------

    st.subheader("Class Probabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Malignant**")

        st.metric(
            "Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

        st.progress(float(probabilities[0]))


    with col2:

        st.write("**Benign**")

        st.metric(
            "Probability",
            f"{probabilities[1] * 100:.2f}%"
        )

        st.progress(float(probabilities[1]))


# --------------------------------------------------
# About the project
# --------------------------------------------------

st.divider()

st.header("📖 About This Project")

st.write(
    """
This project investigates whether a machine learning system
can recognize when it should not automatically trust its own
prediction.

The study evaluates prediction confidence, calibration,
distribution shift, confidence-based abstention, and
robustness across different train-test splits.

When model confidence is below the selected reliability
threshold, the system abstains from automatically accepting
the prediction and flags it for human review.
"""
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "LearnDepth Academy — Track 2 Advanced ML Internship | "
    "Project ML-T2-016"
)