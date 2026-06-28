import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import kagglehub

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
print("[Pipeline] Downloading data from Kaggle...")
download_path = kagglehub.dataset_download("beatafaron/telco-customer-churn-realistic-customer-feedback")
csv_file = [f for f in os.listdir(download_path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(download_path, csv_file))

# Standardize column names to lowercase to prevent case-sensitivity bugs
df.columns = df.columns.str.lower()

# Map target text to binary
if df['churn'].dtype == 'O':
    df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# Enforce numerical conversions safely
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df['monthlycharges'] = pd.to_numeric(df['monthlycharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

# Fill missing data globally before partitioning splits
df['customerfeedback'] = df['customerfeedback'].fillna("No feedback provided")
numerical_cols = ['tenure', 'monthlycharges', 'totalcharges']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Outlier handling via clipping limits
def handle_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    return col.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

for col in numerical_cols:
    df[col] = handle_outliers(df[col])

# Split datasets first to fully guarantee ZERO data leakage during transformations
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df, df['churn'], test_size=0.2, random_state=42, stratify=df['churn']
)

# ==========================================
# 2. FEATURE ENGINEERING (Fitted ONLY on Train Split)
# ==========================================
print("\n[NLP Pipeline] Vectorizing customer feedback text...")
tfidf = TfidfVectorizer(max_features=30, stop_words='english')
tfidf_train = tfidf.fit_transform(X_train_raw['customerfeedback']).toarray()
tfidf_test = tfidf.transform(X_test_raw['customerfeedback']).toarray()

feature_names = [f"text_{word}" for word in tfidf.get_feature_names_out()]
tfidf_train_df = pd.DataFrame(tfidf_train, columns=feature_names, index=X_train_raw.index)
tfidf_test_df = pd.DataFrame(tfidf_test, columns=feature_names, index=X_test_raw.index)

print("[Clustering Pipeline] Isolating customer behavior segments...")
scaler_behavior = StandardScaler()
behavior_train_scaled = scaler_behavior.fit_transform(X_train_raw[['tenure', 'monthlycharges']])
behavior_test_scaled = scaler_behavior.transform(X_test_raw[['tenure', 'monthlycharges']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X_train_raw = X_train_raw.copy()
X_test_raw = X_test_raw.copy()
X_train_raw['behavior_cluster'] = kmeans.fit_predict(behavior_train_scaled)
X_test_raw['behavior_cluster'] = kmeans.predict(behavior_test_scaled)

# Merge datasets cleanly using indexes to maintain spatial integrity
X_train_all = pd.concat([X_train_raw[numerical_cols + ['behavior_cluster']], tfidf_train_df], axis=1)
X_test_all = pd.concat([X_test_raw[numerical_cols + ['behavior_cluster']], tfidf_test_df], axis=1)

# Apply global feature scales safely
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_all)
X_test_scaled = scaler_final.transform(X_test_all)

# Retain original structural strings as DataFrames to clear standard Scikit-Learn validation alerts
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_all.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_all.columns)

# ==========================================
# 3. MODEL BENCHMARKING
# ==========================================
print("\n[Model Benchmarking] Running Cross-Validation tests...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled_df, y_train, cv=3, scoring='accuracy')
    cv_results[name] = scores.mean()
    print(f" -> {name} CV Stability Score: {scores.mean():.4f}")

best_model_name = max(cv_results, key=cv_results.get)
best_model_instance = models[best_model_name]
print(f"\n[Optimization] Selected architecture for optimization: {best_model_name}")

# ==========================================
# 4. HYPERPARAMETER OPTIMIZATION
# ==========================================
if best_model_name == 'Random Forest':
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
elif best_model_name == 'Decision Tree':
    param_grid = {'max_depth': [5, 10, 20]}
elif best_model_name == 'SVM':
    param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
else:
    param_grid = {'C': [0.1, 1, 10]}

grid_search = GridSearchCV(best_model_instance, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled_df, y_train)
optimized_model = grid_search.best_estimator_

# Finalize training fit parameters
optimized_model.fit(X_train_scaled_df, y_train)
y_pred = optimized_model.predict(X_test_scaled_df)
y_prob = optimized_model.predict_proba(X_test_scaled_df)[:, 1]

# Save tracking configurations
joblib.dump(optimized_model, 'best_churn_model.pkl')
joblib.dump(scaler_final, 'final_scaler.pkl')
joblib.dump(scaler_behavior, 'behavior_scaler.pkl')
joblib.dump(kmeans, 'kmeans_behavior_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# ==========================================
# 5. MODEL METRICS REPORT
# ==========================================
print("\n--- Evaluation Metric Metrics Run Summary ---")
print(classification_report(y_test, y_pred))

# ==========================================
# 6. INTERACTIVE PRODUCTION PREDICTOR
# ==========================================
print("\n--- Production Simulation Interface ---")
while True:
    cmd = input("\nEnter to evaluate profile / 'exit' to terminate: ").strip().lower()
    if cmd == 'exit':
        break
    try:
        tenure = float(input("Customer Tenure (months): "))
        monthly = float(input("Monthly Charge Amount ($): "))
        total = float(input("Total Account Spending ($): "))
        feedback = input("Raw text feedback entry: ")

        # Deploy Pipeline Framework sequentially
        behavior_input_df = pd.DataFrame([[tenure, monthly]], columns=['tenure', 'monthlycharges'])
        b_scaled = scaler_behavior.transform(behavior_input_df)
        cluster_id = kmeans.predict(b_scaled)[0]

        text_vec = tfidf.transform([feedback]).toarray()
        text_df = pd.DataFrame(text_vec, columns=feature_names)

        num_df = pd.DataFrame([[tenure, monthly, total, cluster_id]], columns=numerical_cols + ['behavior_cluster'])
        full_row_df = pd.concat([num_df, text_df], axis=1)

        final_scaled_df = pd.DataFrame(final_scaled, columns=full_row_df.columns)
        risk_pred = optimized_model.predict(final_scaled_df)[0]
        risk_prob = optimized_model.predict_proba(final_scaled_df)[0][1] * 100

        print(f"\n[Classification]: {'⚠️ HIGH RISK OF CHURN' if risk_pred == 1 else '✅ STATUS SECURE (RETENTION HIGH)'}")
        print(f"[Probability Breakdown]: Risk Factor at {risk_prob:.2f}%")
        print(f"[Assigned Customer Tier]: Cluster {cluster_id}")
    except Exception as err:
        print(f"Processing error. Verify input parameters. Details: {err}")