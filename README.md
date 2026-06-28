# Hybrid Customer Churn Prediction Pipeline

An end-to-end machine learning system designed to predict telecom customer churn by blending structural customer metrics (unsupervised clustering) with raw customer text feedback (NLP processing). 

## 🚀 Key Features
- **Data Leakage Guards:** Strict split-before-transformation design to eliminate training bias.
- **NLP Engineering:** Text feedback conversion via TF-IDF Vectorization.
- **Customer Segmentation:** K-Means Clustering to group customers by spend and tenure behavior.
- **Automated Benchmarking:** Cross-validated evaluation across Logistic Regression, Decision Trees, Random Forest, and SVM models.
- **Interactive Production Interface:** Real-time console simulation to evaluate instant risk profiles.

## 📁 Repository Structure
- `churn_pipeline.py`: The main automated machine learning script.
- `README.md`: Project overview and explanation.

## 🛠️ How to Run
1. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib kagglehub
