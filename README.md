# 🧠 Stemeta Internship — ML Projects Portfolio

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=flat&logo=pandas)
![Flask](https://img.shields.io/badge/Flask-API%20Deployment-black?style=flat&logo=flask)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

> 🏢 Projects developed during **AI/ML Internship at Stemeta.ai, Islamabad**
> Covering real-world classification problems across healthcare, finance,
> telecom, and HR domains — with full EDA, feature engineering,
> model training, evaluation, and API deployment.

---

## 📁 Projects Overview

| # | Project | Domain | Algorithm | Key Technique |
|---|---------|--------|-----------|---------------|
| 1 | 🔬 Breast Cancer Detection | Healthcare | Random Forest, SVM | Binary Classification |
| 2 | 💳 Credit Card Fraud Detection | Finance | Random Forest, Logistic Regression | SMOTE — Imbalanced Data |
| 3 | 💡 Health Indicator Analysis | Healthcare | ML Classification | EDA + Patient Profiling |
| 4 | ❤️ Heart Failure Prediction | Healthcare | Logistic Regression, XGBoost | Precision-Recall Optimization |
| 5 | 🧑‍💼 IBM HR Employee Attrition | HR Analytics | Decision Tree, Random Forest | Attrition Factor Analysis |
| 6 | 📧 Spam Email Detection | NLP | Multinomial Naive Bayes | TF-IDF Text Vectorization |
| 7 | 📱 Telco Customer Churn | Telecom | Logistic Regression, Decision Tree | Churn Prediction |
| 8 | 🚢 Titanic Survival API | Classic ML | Random Forest | Flask REST API Deployment |
| 9 | ⚖️ Imbalanced Data — Churn | Telecom | Multiple Models | SMOTE, Under/Over Sampling |

---

## 🔍 Project Details

### 1. 🔬 Breast Cancer Detection
Supervised classification model predicting whether a tumor is **benign or malignant**
based on diagnostic imaging features.
- **Dataset:** Wisconsin Breast Cancer Dataset
- **Models:** Random Forest, SVM, Logistic Regression
- **Focus:** High recall to minimize false negatives (missed cancers)

---

### 2. 💳 Credit Card Fraud Detection
Detecting fraudulent transactions in a **highly imbalanced dataset** where
fraud cases are less than 0.2% of all transactions.
- **Challenge:** Extreme class imbalance
- **Solution:** SMOTE oversampling + undersampling techniques
- **Models:** Random Forest, Logistic Regression
- **Metrics:** Precision, Recall, F1, ROC-AUC

---

### 3. 💡 Health Indicator Analysis
Exploratory analysis of general health data to identify key disease
indicators and support healthcare decision-making.
- **Focus:** EDA, feature correlation, patient profiling
- **Techniques:** Univariate & multivariate analysis, visualization

---

### 4. ❤️ Heart Failure Prediction
Predicting likelihood of heart failure from clinical records — where
**precision and recall are critical** for life-saving decisions.
- **Dataset:** Heart Failure Clinical Records Dataset
- **Models:** XGBoost, Logistic Regression, Random Forest
- **Focus:** Optimizing recall — missing a positive case is costly

---

### 5. 🧑‍💼 IBM HR Analytics — Employee Attrition
Predicting which employees are likely to leave the organization and
identifying the key factors driving attrition.
- **Dataset:** IBM HR Analytics Dataset (1,470 employees)
- **Models:** Decision Tree, Random Forest
- **Insights:** Feature importance analysis for HR decision support

---

### 6. 📧 Spam Email Detection
NLP-based binary text classification system to detect spam emails
with high accuracy using classical NLP techniques.
- **Technique:** TF-IDF Vectorization
- **Model:** Multinomial Naive Bayes
- **Libraries:** NLTK, Scikit-learn
- **Focus:** Text preprocessing, stop word removal, stemming

---

### 7. 📱 Telco Customer Churn Prediction
Predicting customer churn for a telecom company to help retention
teams proactively target at-risk customers.
- **Dataset:** IBM Telco Customer Churn Dataset
- **Models:** Logistic Regression, Decision Tree
- **Features:** Demographics, account info, service usage data

---

### 8. 🚢 Titanic Survival Prediction API
End-to-end ML project with **REST API deployment** — predicting
passenger survival and serving predictions via Flask.
- **Model:** Random Forest Classifier
- **Deployment:** Flask REST API
- **Input:** Passenger features → **Output:** Survival prediction (JSON)

```bash
# Example API call
POST /predict
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 28,
  "SibSp": 0,
  "Fare": 100
}
# Response: {"survived": 1, "probability": 0.92}
```

---

### 9. ⚖️ Handling Imbalanced Data — Customer Churn
Deep-dive study into resolving **class imbalance** in the Telco Churn
dataset and measuring its impact on model performance.
- **Techniques Compared:**
  - SMOTE (Synthetic Minority Oversampling)
  - Random Oversampling
  - Random Undersampling
  - Combined Approach
- **Finding:** SMOTE + Random Forest gave best F1 score

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| ML Library | Scikit-learn |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK, TF-IDF |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| API Deployment | Flask |
| Notebooks | Jupyter Notebook |
| Version Control | Git, GitHub |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/tashfeen786/STEMETA_Intership_Projects.git
cd STEMETA_Intership_Projects

# Install dependencies
pip install -r requirements.txt

# Open any project notebook
jupyter notebook
```

---

## 🏗️ Project Structure

```
STEMETA_Intership_Projects/
│
├── breast_cancer/                    # Breast cancer classification
├── Credit Card Fraud Detection/      # Fraud detection + SMOTE
├── HealthIndicator/                  # Health data EDA & modeling
├── Heart_Failure_Prediction/         # Heart failure risk prediction
├── IBM HR Analytics.../              # Employee attrition analysis
├── Spam_email_Detection/             # NLP spam classifier
├── Telco_Customer_Churn/             # Customer churn prediction
├── TitanicAPI/                       # Flask API deployment
├── Handling_imbalanced_data_...ipynb # Imbalanced data study
└── requirements.txt                  # Python dependencies
```

---

## 📈 Key Learnings from Internship

- ✅ Real-world data is messy — EDA and preprocessing matter most
- ✅ Class imbalance is a critical challenge in production ML
- ✅ Model selection depends on domain — healthcare needs high recall
- ✅ Deployment bridges the gap between ML model and real product
- ✅ Feature engineering often matters more than model choice

---

## 👨‍💻 Author

**Tashfeen Aziz** — AI/ML Engineer & Python Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/tashfeen-aziz-b51361292)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/tashfeen786)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:tashfeen247@gmail.com)

---

⭐ **If you found these projects helpful, please give it a star!**

*Built during AI/ML Internship at Stemeta.ai, Islamabad 🇵🇰*
