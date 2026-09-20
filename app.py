import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import shap

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="BNPL Privacy Risk Detector",
    page_icon="🔐",
    layout="centered"
)

st.title("🔐 BNPL Privacy Risk Detection")
st.caption(
    "Machine Learning based privacy-risk assessment prototype"
)

# -----------------------------
# Features
# -----------------------------
features = [
    "permissions_count",
    "data_collection_level",
    "third_party_sharing",
    "advertising_tracking",
    "encryption",
    "mfa",
    "secure_server",
    "data_deletion_option"
]

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/bnpl_privacy_dataset.csv")

target_map = {
    "High": 0,
    "Medium": 1,
    "Low": 2
}

inverse_map = {
    0: "High",
    1: "Medium",
    2: "Low"
}

X = df[features]
y = df["risk_level"].map(target_map)

# -----------------------------
# Train XGBoost Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Model accuracy
test_prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, test_prediction)

# -----------------------------
# Model Information
# -----------------------------
with st.expander("Model Information"):
    st.write("**Algorithm:** XGBoost Classifier")
    st.write(f"**Test Accuracy:** {accuracy * 100:.2f}%")
    st.write("**Training samples:**", len(X_train))
    st.write("**Testing samples:**", len(X_test))

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Privacy-Related Factors")

permissions = st.slider(
    "Number of permissions",
    2, 15, 7
)

collection = st.selectbox(
    "Data collection level",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Low",
        2: "Medium",
        3: "High"
    }[x]
)

third_party = st.selectbox(
    "Third-party data sharing",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

tracking = st.selectbox(
    "Advertising / tracking",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

encryption = st.selectbox(
    "Encryption",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

mfa = st.selectbox(
    "Multi-factor authentication (MFA)",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

secure = st.selectbox(
    "Secure server usage",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

deletion = st.selectbox(
    "Data deletion option",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Privacy Risk", type="primary"):

    row = pd.DataFrame([[
        permissions,
        collection,
        third_party,
        tracking,
        encryption,
        mfa,
        secure,
        deletion
    ]], columns=features)

    # Prediction
    pred_num = int(model.predict(row)[0])
    risk = inverse_map[pred_num]

    # Probability
    probabilities = model.predict_proba(row)[0]
    confidence = probabilities[pred_num] * 100

    # -------------------------
    # Privacy Score
    # -------------------------
    score = 100

    score -= permissions * 2.5
    score -= collection * 7
    score -= third_party * 10
    score -= tracking * 6

    score += encryption * 9
    score += mfa * 8
    score += secure * 6
    score += deletion * 7

    score = max(0, min(100, round(score, 1)))

    category = {
        "High": "Privacy Risky",
        "Medium": "Moderately Secure",
        "Low": "Highly Secure"
    }[risk]

    # -------------------------
    # Results
    # -------------------------
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Privacy Score",
            f"{score}/100"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk
        )

    st.info(
        f"Privacy Category: **{category}**"
    )

    st.progress(
        min(score / 100, 1.0)
    )

    st.write(
        f"**Model confidence:** {confidence:.2f}%"
    )

    # -------------------------
    # Model Feature Importance
    # -------------------------
    st.subheader("📊 Model Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    st.bar_chart(importance)

    # -------------------------
    # SHAP Explanation
    # -------------------------
    st.subheader("🧠 Explainable AI — SHAP")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)

    # Handle different SHAP versions / multiclass output
    if isinstance(shap_values, list):
        local_values = shap_values[pred_num][0]
    elif len(shap_values.shape) == 3:
        local_values = shap_values[0, :, pred_num]
    else:
        local_values = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": features,
        "SHAP Impact": local_values
    })

    explanation["Absolute Impact"] = (
        explanation["SHAP Impact"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )

    st.write(
        "Features are ranked according to their contribution "
        "to the model prediction."
    )

    st.dataframe(
        explanation[
            ["Feature", "SHAP Impact"]
        ],
        use_container_width=True
    )

    # -------------------------
    # Risk Interpretation
    # -------------------------
    st.subheader("📌 Interpretation")

    if risk == "High":
        st.error(
            "The model predicts a high privacy-risk level "
            "based on the supplied factors."
        )

    elif risk == "Medium":
        st.warning(
            "The model predicts a medium privacy-risk level "
            "based on the supplied factors."
        )

    else:
        st.success(
            "The model predicts a low privacy-risk level "
            "based on the supplied factors."
        )

# -----------------------------
# Disclaimer
# -----------------------------
st.warning(
    "This is an academic prototype using a synthetic dataset. "
    "The prediction should not be treated as a real security, "
    "privacy, financial, or compliance assessment of a BNPL provider."
)