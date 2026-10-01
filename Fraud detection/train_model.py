import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/fraud_data.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Remove empty rows
df = df.dropna()

# Find target column
if "isFraud" in df.columns:
    target = "isFraud"
elif "is_fraud" in df.columns:
    target = "is_fraud"
else:
    raise ValueError(
        "Fraud column not found. Available columns: "
        + str(df.columns.tolist())
    )

# Separate input and target
X = df.drop(columns=[target])
y = df[target]

# Remove ID column if present
for col in ["id", "ID", "transaction_id", "TransactionID"]:
    if col in X.columns:
        X = X.drop(columns=[col])

# Convert categorical data to numbers
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("FINANCIAL FRAUD DETECTION")
print("==============================")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and column names
with open("fraud_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("\nModel saved as fraud_model.pkl")