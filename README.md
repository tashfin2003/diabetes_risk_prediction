# Diabetes Risk Classification using Machine Learning

## Project Overview

This project develops and evaluates machine learning classification
models to predict **diabetes risk levels**:

- **Low**
- **Moderate**
- **High**

Three classification algorithms were trained and compared:

1.  Logistic Regression
2.  Random Forest
3.  Gradient Boosting

The models were evaluated using Accuracy, Precision, Recall, F1-score,
and Confusion Matrix.



## Dataset

- **Total samples:** 15,000
- **Training samples:** 12,000
- **Testing samples:** 3,000
- **Predictive features after preprocessing:** 15 selected features
- **Target:** `diabetes_risk`

### Target Distribution

| Risk Level | Proportion |
|------------|-----------:|
| Low        |        60% |
| Moderate   |        25% |
| High       |        15% |

Target encoding:

``` text
Low       → 0
Moderate  → 1
High      → 2
```



## Data Preprocessing

- Removed `patient_id`
- Encoded the target variable
- Encoded binary and ordinal categorical variables
- Applied one-hot encoding to nominal categorical features
- Handled missing values using the training-set mode
- Applied IQR-based outlier capping
- Created Mean Arterial Pressure (MAP) and BMI Category features
- Selected the top 15 features using Random Forest feature importance
- Applied StandardScaler for Logistic Regression
- Used an 80/20 stratified train-test split

Preprocessing parameters were learned from the training data and then
applied to the test data to avoid data leakage.



## Models

### Logistic Regression

Trained using standardized features.

### Random Forest

``` text
n_estimators = 200
random_state = 42
```

Feature scaling was not required because Random Forest is tree-based.

### Gradient Boosting

Trained using the preprocessed training features.



## Results

The models were evaluated on **3,000 unseen test samples**.

### Overall Performance

| Model               |   Accuracy | Macro F1 | Weighted F1 |
|---------------------|-----------:|---------:|------------:|
| Logistic Regression | **78.63%** |     0.73 |        0.78 |
| Random Forest       |     78.20% |     0.73 |        0.78 |
| Gradient Boosting   |     78.60% | **0.74** |        0.78 |

### Class-wise Performance

| Model               | Class    | Precision | Recall | F1-score |
|---------------------|----------|----------:|-------:|---------:|
| Logistic Regression | Low      |      0.86 |   0.90 |     0.88 |
| Logistic Regression | Moderate |      0.59 |   0.56 |     0.57 |
| Logistic Regression | High     |      0.81 |   0.69 |     0.74 |
| Random Forest       | Low      |      0.85 |   0.90 |     0.88 |
| Random Forest       | Moderate |      0.58 |   0.55 |     0.57 |
| Random Forest       | High     |      0.82 |   0.68 |     0.74 |
| Gradient Boosting   | Low      |      0.86 |   0.89 |     0.88 |
| Gradient Boosting   | Moderate |      0.58 |   0.58 |     0.58 |
| Gradient Boosting   | High     |      0.82 |   0.70 | **0.75** |



## Findings

- All three models achieved approximately **78% overall accuracy**.
- The **Low-risk** class was classified most accurately, with an
  F1-score of approximately **0.88** across all models.
- The **Moderate-risk** class was the most difficult to classify, with
  F1-scores between **0.57 and 0.58**.
- A major source of error was classifying **Moderate-risk cases as
  Low-risk**.
- Another significant error was classifying **High-risk cases as
  Moderate-risk**.
- Gradient Boosting achieved the highest **Macro F1 (0.74)** and the
  highest **High-class F1 (0.75)**.
- Logistic Regression achieved the highest overall accuracy at
  **78.63%**, while Gradient Boosting achieved a very similar accuracy
  of **78.60%**.
- Because the dataset is imbalanced, accuracy alone does not fully
  describe model performance. Class-wise Precision, Recall, F1-score,
  and Macro F1 were also considered.



## Project Workflow

``` text
Dataset
   ↓
Data Inspection
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Outlier Capping
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Confusion Matrix & Findings
```



## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab



## Conclusion

The experiment shows that machine learning can classify diabetes risk
into Low, Moderate, and High categories with approximately **78%
accuracy** on the test dataset.

The main challenge was distinguishing the **Moderate-risk** class from
the Low- and High-risk classes. The results also demonstrate the
importance of using multiple evaluation metrics rather than relying only
on accuracy for an imbalanced classification problem.



## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Improved feature engineering
- Alternative class-imbalance techniques
- Testing additional boosting algorithms
- Further feature selection experiments
- Evaluation using the official competition metric



## Author

**Tawhidul Hoque Tashfin**

Machine Learning / Datathon Project
