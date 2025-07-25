## 🧬 🩺 Chronic Kidney Disease Prediction (CKDP)

## 📌 Project Overview

Chronic Kidney Disease (CKD) affects 1 in 10 people globally, often going undetected until it's too late. Chronic Kidney Disease (CKD) is a long-term condition where the kidneys do not work as well as they should. It is a growing public health issue affecting millions globally. Early detection is crucial to managing and treating CKD effectively. However, due to the complexity and volume of clinical data, manual diagnosis is often time-consuming and prone to human error.

This project presents an end-to-end Machine Learning solution that leverages patient data to predict the presence of CKD. The goal is to demonstrate how artificial intelligence can assist healthcare professionals in making quicker and more accurate decisions. The project covers everything from data preprocessing and exploratory data analysis (EDA) to model building, evaluation, and predictions on unseen data.

---

## 🧠 What This Project Does

🔹 **Inputs**: Patient lab records (e.g., hemoglobin, creatinine, blood pressure)  
🔹 **Outputs**: Predicts the likelihood of Chronic Kidney Disease  
🔹 **Goal**: Assist early detection and reduce risk through predictive insights

This notebook turns complex health data into actionable intelligence using advanced machine learning techniques. By automating diagnosis with >98% accuracy, it enables faster, smarter, and more reliable clinical decisions.

---

## 📊 Real-World Dataset

- **Source**: Kaggle Repository
- **Size**: 400 patient records
- **Features**: 24 attributes including symptoms, lab values, and risk indicators
- **Target**: `classification` (1 = CKD, 0 = Not CKD)

### Examples of key features:
- Hemoglobin (hemo)
- Blood Urea (bu)
- Serum Creatinine (sc)
- Diabetes Mellitus (dm)
- Red Blood Cell Count (rbc)
- Hypertension (htn)

---

## 🧪 ML Pipeline Breakdown

### 1. 🧹 Data Cleaning & Preparation
- Handled missing values using imputation strategies (mean/mode/fill-forward)
- Removed extreme outliers using IQR
- Encoded categorical features (Label)

### 2. 📊 EDA & Insights
- Used Seaborn and Matplotlib to explore trends
- Found strong correlation between low hemoglobin + CKD
- Patients with diabetes & hypertension were more likely to have CKD

### 3. ⚙️ Feature Engineering & Scaling
- Standardized numerical features with `StandardScaler`
- Selected high-impact features using Chi-Square and correlation analysis

### 4. 🤖 Model Training
Trained and compared 5 supervised models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier 🌟 (Best Model)
- XG Boost Classifier 🌟 (Best Model)
- Ada Boost 
- Support Vector Machine (SVM)
- Naive Bayes

### 5. 🧪 Evaluation Metrics
Used classification metrics including:
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC

```📈 Results

| Model                      | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression       | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| Random Forest Classifier  | 0.99     | 0.97      | 1.00   | 0.99     | 1.00    |
| Support Vector Machine    | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| K-Nearest Neighbors (KNN) | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| Naive Bayes               | 0.94     | 1.00      | 0.90   | 0.95     | 1.00    |
| AdaBoost Classifier       | 0.99     | 1.00      | 0.97   | 0.99     | 0.99    |
| XGBoost Classifier        | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |

```

---

## 🧠 Why It Matters

> “Data is the new stethoscope.”

CKD is a silent killer. This model demonstrates how data science can save lives by:
- Supporting doctors with instant risk prediction
- Standardizing decisions using interpretable models
- Encouraging early diagnosis, especially in rural/low-resource regions

---

## 💻 Tech Stack

| Layer        | Tools/Libs Used                   |
|--------------|----------------------------------|
| Language     | Python 3.10+                     |
| Libraries    | Pandas, NumPy, Matplotlib, Seaborn |
| ML Framework | scikit-learn                     |
| IDE          | Jupyter Notebook                 |
| Deployment   | (Future scope: Flask/Streamlit)  |

---

## 🧭 How to Run This Notebook

1. Clone the repo:
```bash
git clone https://github.com/yourusername/ckdp.git
cd ckdp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the notebook:
```bash
jupyter notebook ckdp.ipynb
```

4. Run all cells sequentially.

---

## 🧾 Conclusion

This CKD prediction model combines domain knowledge with powerful AI techniques to support life-saving early detection. The model is robust, interpretable, and tested on real-world clinical data — a strong foundation for future healthtech applications.

---

## 👤 Developed By

**Ari R.**  
_Data Scientist_  
🔗 [GitHub](https://github.com/ari-r-1) | 📧 ariranalyst@gmail.com

---

## ⚖️ License

This project is licensed under the **MIT License**.

---

🩺 *"Prevention is better than dialysis." — Predict CKD before it's too late.*
