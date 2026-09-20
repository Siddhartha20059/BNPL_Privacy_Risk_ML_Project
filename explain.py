import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/bnpl_privacy_dataset.csv")

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

X = df[features]

label_map = {
    "High": 0,
    "Medium": 1,
    "Low": 2
}

y = df["risk_level"].map(label_map)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature importance
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=features,
    show=False
)

plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300)
plt.show()

print("SHAP analysis completed.")
print("Saved: shap_feature_importance.png")