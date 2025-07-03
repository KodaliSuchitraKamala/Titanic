# Titanic

A data-science project predicting passenger survival on the RMS Titanic using machine learning and comprehensive analysis.

---

## 📚 Dataset

- **Source**: Kaggle's Titanic dataset
- **Train set**: `train.csv` — 891 rows, 12 columns
- **Test set**: `test.csv` — 418 rows, 11 columns
- **Key features**:
  - `Survived` (target): 0 = No, 1 = Yes  
  - `Pclass` – Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)  
  - `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

---

## 🔍 Exploratory Data Analysis (EDA)

Performed in `Titanic.ipynb`. Highlights include:

- Survival trends across class, gender, age, family, and embarkation point
- Visualizations using Seaborn & Matplotlib
- Data insights:
  - Higher survival for **females** and **1st-class** passengers
  - Family and age-based survival variations
  - Embarkation point impact on survival

---

## 🛠 Feature Engineering

Key transformations:

1. **Missing values**: Imputed `Age` and `Embarked`.
2. **Title extraction**: Created `Title` from `Name` (Mr, Miss, etc.).
3. **Cabin simplification**: Used cabin deck letter.
4. **Family size**: Combined `SibSp` + `Parch` + 1.
5. **Categorical conversions**: Age and Fare into bins.
6. **Label encoding**: Gender, Title, Cabin, Embarked.

---

## 🧠 Model Training & Evaluation

- Split train data into features (`X`) and target (`y`)
- Evaluated models with 10-fold cross-validation:
  - KNN, Decision Trees, Random Forests, SVC, Gradient Boosting, etc.
  - Best performance: **Support Vector Classifier** ~ 83.5% accuracy
- The notebook includes training code and model comparisons

---

## 🧩 Content Files

- `Titanic.ipynb`: Full EDA, feature engineering, modeling flow
- `train.csv`, `test.csv`: Original data files
- `model.pkl`: Serialized final model
- `README.md`: Project overview (this file)

---

## 🚀 How to Run

1. **Prereqs**: Python 3.8+, pip
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
