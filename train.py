import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Convert target labels to numbers
label_map = {
    "High": 0,
    "Medium": 1,
    "Low": 2
}

y = df["risk_level"].map(label_map)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss"
    )
}

results = []

# Train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4)
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["High", "Medium", "Low"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# Model comparison
print("\n\nMODEL COMPARISON")
print("=" * 50)

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))