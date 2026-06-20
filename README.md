# 🏢 Employee Attrition Prediction using Machine Learning

Predicting employee attrition is a crucial challenge for organizations, as high turnover leads to increased recruitment costs, lost productivity, and disrupted operations. This repository contains an end-to-end Machine Learning project designed to analyze employee demographics, tenure, and job-related factors to predict whether an employee will stay (Active) or leave (Terminated).

The project is structured as a step-by-step pipeline across a series of Jupyter Notebooks, moving from initial Data Exploration to Preprocessing, Feature Engineering, Model Training, and Threshold Optimization.

---

## 📁 Repository Structure

```text
├── data/
│   ├── MFG10YearTerminationData.csv      # Raw dataset (49,653 rows)
│   └── cleaned_employee_attrition.csv   # Preprocessed dataset
├── notebooks/
│   ├── 01_data_understanding.ipynb       # Exploratory Data Analysis (EDA)
│   ├── 02_data_cleaning.ipynb            # Handling missing values and data leakage
│   ├── 03_Feature_Engineering.ipynb       # One-hot encoding & train-test splitting
│   ├── 04_model_training_and_evaluation.ipynb # Baseline Logistic Regression model
│   └── 05_RandomForest_and_GradientBoostingCLassifier.ipynb # Advanced ensemble models & threshold tuning
└── README.md                             # Project documentation
```

---

## 📊 Dataset Overview

The dataset used is `MFG10YearTerminationData.csv` (representing 10 years of employee records from a manufacturing organization). It contains **49,653 observations** and **18 columns**.

### Feature Details:
*   **Target Variable**:
    *   `STATUS`: The employee's status (`ACTIVE` or `TERMINATED`).
*   **Demographics**:
    *   `age`: Age of the employee.
    *   `gender_full` / `gender_short`: Gender (`Male` / `Female`).
    *   `birthdate_key`: Date of birth.
*   **Employment Details**:
    *   `length_of_service`: Tenure in years.
    *   `orighiredate_key`: Original date of hire.
    *   `terminationdate_key`: Date of termination (defaults to `1/1/1900` for active employees).
    *   `city_name`: Employee's location.
    *   `department_name`: Department (e.g., Executive, Produce, Meats, Bakery).
    *   `job_title`: Job role (e.g., Cashier, Meat Cutter, Produce Clerk).
    *   `store_name`: Unique store ID.
    *   `BUSINESS_UNIT`: `STORES` or `HEADOFFICE`.
*   **Post-Hoc / Leakage Features**:
    *   `termreason_desc`: Reason for leaving (e.g., Retirement, Resignation, Layoff).
    *   `termtype_desc`: Termination type (e.g., Voluntary, Involuntary).

---

## 🛠️ Machine Learning Pipeline

### 1. Data Understanding & Exploratory Analysis
*   Identified the dataset shape: **49,653 entries, 18 columns**.
*   Checked for missing values: The dataset has **zero null values**.
*   Identified class imbalance: **48,168 Active (0)** and **1,485 Terminated (1)** (~3% attrition rate).

### 2. Data Cleaning & Data Leakage Prevention
*   **Data Leakage Warning**: `termreason_desc` and `termtype_desc` were dropped because they represent information known only *after* an employee leaves. Keeping them would cause model overfitting.
*   Redundant feature `gender_short` was dropped (retaining `gender_full`).
*   Mapped target column `STATUS` to binary (`ACTIVE` -> `0`, `TERMINATED` -> `1`).
*   Identified that IDs (`EmployeeID`) and raw dates (`birthdate_key`, `orighiredate_key`, `terminationdate_key`, `recorddate_key`) must be dropped or transformed to ensure models generalize to new time periods.

### 3. Feature Engineering
*   Applied One-Hot Encoding via `pd.get_dummies` for categorical columns: `city_name`, `department_name`, `job_title`, `gender_full`, and `BUSINESS_UNIT`.
*   Removed specific timestamps and transaction IDs to avoid target leakage across years.
*   Split the dataset into an **80% Training Set** and **20% Test Set** using a stratified split to preserve the 3% attrition ratio.

### 4. Model Training & Tuning
Multiple algorithms were trained, evaluated, and adjusted to tackle the high class imbalance:
*   **Logistic Regression**: Used as a baseline model. Explored default vs. balanced class weights.
*   **Random Forest Classifier**: Built ensemble models with standard and balanced class weights.
*   **Extra Trees Classifier**: Evaluated as a variant of bagging ensembles.
*   **Gradient Boosting Classifier**: Trained to capture complex non-linear interactions.
*   **Threshold Tuning**: Adjusted the classification probability threshold on the best model to maximize the F1-score for the minority class (`Terminated`).

---

## 📈 Model Performance & Comparison

Due to the dataset's extreme class imbalance, focusing purely on Accuracy is misleading. Precision, Recall, and F1-Score for the **Terminated class (1)** are the primary target metrics:

| Model | Overall Accuracy | Attrition Precision | Attrition Recall | Attrition F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Default)** | 96.9% | 39.0% | 2.0% | 4.0% | - |
| **Logistic Regression (Balanced / Tuned)** | 94.0% | 24.0% | 51.0% | 32.0% | 0.8778 |
| **Random Forest (No Dates)** | 98.6% | 90.0% | 59.0% | 71.0% | - |
| **Random Forest (Balanced / No Dates)** | 97.0% | 45.0% | 66.0% | 54.0% | - |
| **Extra Trees Classifier** | 98.0% | 80.0% | 50.0% | 62.0% | - |
| **Gradient Boosting (Default, Threshold = 0.50)** | **99.0%** | **99.0%** | 55.0% | 71.0% | - |
| **Gradient Boosting (Tuned, Threshold = 0.30)** | **98.0%** | 71.0% | **68.0%** | **70.0%** | - |

### 💡 Key Takeaway:
The **Gradient Boosting Classifier with a tuned decision threshold of 0.30** provides the best balance for business deployment. It successfully captures **68% of employees who will leave** (Recall) while maintaining a **71% precision rate** (minimizing false positives, which could lead to wasting retention budgets on employees who intend to stay).

---

## 🔑 Key Insights & Feature Importances

Analysis of the Random Forest model's feature importances revealed the top drivers of employee attrition:

1.  **Age (60.3% Importance)**: The single most critical driver. Attrition rates correlate strongly with age, showing peaks around retirement ages or early career transitions.
2.  **Length of Service (18.8% Importance)**: Tenure plays a major role in predicting attrition.
3.  **Gender (4.8% Importance)**: Gender (specifically male) shows minor correlation.
4.  **Store Location (3.6% Importance)**: Store locations (e.g., Vancouver, Victoria) play a minor role, representing geographic variations.
5.  **Job Title / Department**: Specific roles like *Produce Clerk* and *Meat Cutter* have distinct attrition profiles compared to executive roles.

---

## 🚀 How to Run the Project

### Prerequisites
Ensure you have Python 3.8+ and the following packages installed:
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `jupyter`

Install dependencies using pip:
```bash
pip install pandas numpy scikit-learn jupyter
```

### Steps
1.  Clone this repository and navigate to the project directory:
    ```bash
    git clone https://github.com/your-username/employee-attrition-ml.git
    cd employee-attrition-ml
    ```
2.  Launch Jupyter Notebooks:
    ```bash
    jupyter notebook
    ```
3.  Run the notebooks in chronological order:
    *   Start with `notebooks/01_data_understanding.ipynb` to explore the raw dataset.
    *   Proceed through `02_data_cleaning.ipynb`, `03_Feature_Engineering.ipynb`, `04_model_training_and_evaluation.ipynb`, and finish with `05_RandomForest_and_GradientBoostingCLassifier.ipynb`.
