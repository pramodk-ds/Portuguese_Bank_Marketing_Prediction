# 📊 Portuguese Bank Marketing Prediction  

A machine learning project to predict whether a client will **subscribe to a term deposit** based on the famous **Portuguese Bank Marketing dataset**.  

---

## 🚀 Project Overview  
This project applies **data preprocessing, feature engineering, and multiple classification models** to solve a real-world business problem.  
The main goal is to assist banks in targeting potential customers more effectively.  

---

## 📂 Dataset  
- **Source**: UCI Machine Learning Repository (Bank Marketing Dataset)  
- **Target Variable**: `y` (Client subscribed to a term deposit? `yes` / `no`)  
- **Features**:  
  - Client attributes: age, job, marital status, education, balance...  
  - Campaign-related: contact type, number of contacts, previous outcome...  
  - Socioeconomic indicators  

---

## ⚙️ Tech Stack  
- **Language**: Python 🐍 (Google Colab)  
- **Libraries**:  
  - `numpy`, `pandas` → Data handling  
  - `matplotlib`, `seaborn` → Visualization  
  - `scikit-learn` → ML models  
  - `imblearn` → Handling class imbalance (SMOTE)  

---

## 🔑 Key Steps  
1. **Exploratory Data Analysis (EDA)**  
   - Visualizations & insights on client behavior  
2. **Data Preprocessing**  
   - Handling missing values, encoding categorical features, scaling  
3. **Class Imbalance Fix**  
   - Applied **SMOTE** to balance target variable  
4. **Model Training**  
   - Trained multiple classification models  
5. **Evaluation**  
   - Confusion Matrix, Precision, Recall, F1-Score, Accuracy  

---

## 🤖 Models & Results  

| Model                   | Test Accuracy |
|--------------------------|---------------|
| Logistic Regression      | **82%** |
| Random Forest Classifier | **89%** |
| Decision Tree Classifier | **84%** |

✅ Best performing model: **Random Forest** with **89% accuracy**.  

---

## ⚔️ Challenges Faced  
- **Imbalanced Dataset**: Most customers did not subscribe (`no`), leading to bias. Fixed using **SMOTE**.  
- **Categorical Variables**: Required encoding for job, education, marital status, etc.  
- **Overfitting in Decision Tree**: Pruned and tuned hyperparameters to improve generalization.  
- **Runtime Issues on Local PC**: Heavy computations were shifted to **Google Colab** with GPU support.  

---

## ▶️ How to Run  
```bash
# Clone this repo
git clone https://github.com/yourusername/Portuguese-Bank-Prediction.git
cd Portuguese-Bank-Prediction

# Install dependencies
pip install -r requirements.txt

# Run on Google Colab (recommended)
Upload Portuguese_Bank_Marketing_Prediction.ipynb to Colab and execute step by step.
```

---

## 🌟 Future Improvements  
- Hyperparameter tuning with GridSearchCV  
- Deploy as a web app using **Flask/Streamlit**  
- Integrate explainability (SHAP / LIME)  

---

## 👨‍💻 Author
**Pramod K**  
Data Science Enthusiast | Machine Learning | Deep Learning

---
