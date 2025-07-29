import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data
df = pd.read_csv('/content/data.tsv', sep='\t')

# Filter target
df = df[df['DEPARTURE_MODE'].isin(['TRUCK', 'TRAIN'])]
y = df['DEPARTURE_MODE'].map({'TRUCK': 0, 'TRAIN': 1})

# Features to use
feature_cols = [
    'FULL_OR_EMPTY_CODE',
    'IS_HAZARD',
    'MAX_TEMPERATURE',
    'ORGANIZATION_LINE_CODE',
    'ORGANIZATION_AGENT_CODE',
    'WEIGHT',
    'CONTAINER_TYPE_CODE'
]

# Fill missing values
X = df[feature_cols].fillna('missing')

# Ensure WEIGHT is numeric
X['WEIGHT'] = pd.to_numeric(X['WEIGHT'], errors='coerce').fillna(0)

# Encode categorical features
X = pd.get_dummies(X, columns=[
    'FULL_OR_EMPTY_CODE',
    'ORGANIZATION_AGENT_CODE',
    'ORGANIZATION_LINE_CODE',
    'CONTAINER_TYPE_CODE'
])

# Check correlations for analysis
df_encoded = X.copy()
df_encoded['target'] = y
correlations = df_encoded.corr()['target'].sort_values()
print("Feature Correlations with Target:")
print(correlations)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RandomForest
print("\nTraining RandomForestClassifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("RandomForest Results:")
print(classification_report(y_test, rf_preds, target_names=['TRUCK', 'TRAIN']))

# Feature importance plot
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importances.sort_values().plot(kind='barh', figsize=(8, 6))
plt.title("RandomForest Feature Importances")
plt.show()

# Confusion matrix
rf_cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['TRUCK', 'TRAIN'], yticklabels=['TRUCK', 'TRAIN'])
plt.title("RandomForest Confusion Matrix")
plt.show()

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# XGBoost
print("\nTraining XGBClassifier...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print("XGBClassifier Results:")
print(classification_report(y_test, xgb_preds, target_names=['TRUCK', 'TRAIN']))

# Feature importance plot
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_importances.sort_values().plot(kind='barh', figsize=(8, 6))
plt.title("XGBoost Feature Importances")
plt.show()

# Confusion matrix
xgb_cm = confusion_matrix(y_test, xgb_preds)
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['TRUCK', 'TRAIN'], yticklabels=['TRUCK', 'TRAIN'])
plt.title("XGBoost Confusion Matrix")
plt.show()

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
