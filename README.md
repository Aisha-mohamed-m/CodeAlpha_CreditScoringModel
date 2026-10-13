# CodeAlpha_CreditScoringModel

Credit scoring model that predicts whether a borrower will default on a loan.
Built for **CodeAlpha Machine Learning Internship, Task 1: Credit Scoring Model**.

## Objective
Predict an individual's creditworthiness (`loan_status`: 1 = default, 0 = no default) from past financial data, using classification algorithms and evaluating them with Precision, Recall, F1-Score and ROC-AUC.

## Dataset
`crd.csv`: 32,581 loans, 12 columns.

| Column | Description |
|---|---|
| `person_age`, `person_income` | Borrower age and yearly income |
| `person_home_ownership` | RENT, MORTGAGE, OWN, OTHER |
| `person_emp_length` | Years of employment |
| `loan_intent`, `loan_grade` | Purpose of the loan and risk grade (A best, G worst) |
| `loan_amnt`, `loan_int_rate`, `loan_percent_income` | Loan amount, interest rate, loan as a share of income |
| `cb_person_default_on_file` | Previous default recorded (Y/N) |
| `cb_person_cred_hist_length` | Length of credit history (years) |
| `loan_status` | **Target** (22% default, 78% no default) |

## Project steps
1. **EDA**: missing values (`person_emp_length` 2.75%, `loan_int_rate` 9.56%), 165 duplicates, class imbalance, distributions, default rate per feature, correlation.
2. **Cleaning**: removed 165 duplicates and 7 impossible rows (age above 100, employment length above 60 years or longer than the age). Valid high incomes were kept, because removing outliers (IQR or 99th percentile) did not improve ROC-AUC.
3. **Feature engineering**:
   - `high_grade` (grade D or worse; default rate jumps from 21% to 59% at D)
   - log transforms of skewed features
   - `grade_x_pct` (loan grade x loan-to-income ratio)
4. **Preprocessing without data leakage**: encode, then stratified 80/20 split, then median imputation and standard scaling fitted on the **training set only**.
5. **Modeling**: Logistic Regression, Decision Tree (max depth 8) and Random Forest (300 trees, `max_depth=12`, `min_samples_leaf=5` to limit overfitting), all with `class_weight="balanced"`.
6. **Evaluation**: 5-fold stratified cross-validation plus a final check on the untouched test set.

## Results

5-fold cross-validation on the training set (mean ± std):

| Model | ROC-AUC | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.885 ± 0.004 | 0.560 ± 0.009 | 0.801 ± 0.009 | 0.659 ± 0.009 | 0.885 |
| Decision Tree | 0.911 ± 0.001 | 0.831 ± 0.030 | 0.735 ± 0.007 | 0.780 ± 0.012 | 0.911 |
| **Random Forest** | **0.930 ± 0.003** | **0.851 ± 0.016** | 0.753 ± 0.003 | **0.799 ± 0.007** | **0.935** |

Random Forest on the test set (threshold 0.5): accuracy 0.92, default precision 0.86, default recall 0.77, F1 0.81.

### Overfitting check (train vs test ROC-AUC)

| Model | Train | Test | Gap |
|---|---|---|---|
| Logistic Regression | 0.886 | 0.885 | 0.001 |
| Decision Tree (depth 8) | 0.921 | 0.911 | 0.010 |
| Random Forest (depth 12, leaf 5) | 0.969 | 0.935 | 0.034 |

An unrestricted Random Forest memorises the training data (train ROC-AUC 1.00, test 0.940). Limiting its depth and leaf size shrinks the gap from 0.060 to 0.034 for a small loss in test score (0.940 to 0.935). Cross-validation scores are stable (std below 0.005) and close to the test scores.

## Key findings
- Default risk rises with the loan grade: A 10%, B 16%, C 21%, D 59%, E 65%, F 71%, G 98%.
- Renters default about four times more often than home owners (32% vs 7%).
- Defaulters have a higher loan-to-income ratio and interest rate, and a lower income.
- Age and credit history length are almost unrelated to default.
- The most important features are `grade_x_pct`, the loan-to-income ratio, the loan grade and income.

## Files
- `CodeAlpha_CreditScoring_final.ipynb`: full notebook (EDA, cleaning, features, models, evaluation)
- `crd.csv`: dataset
- `README.md`: this file

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook CodeAlpha_CreditScoring.ipynb
```
Keep `crd.csv` in the same folder as the notebook.

## Possible improvements
- Tune the decision threshold to raise recall (catch more defaulters) at the cost of precision.
- Try gradient boosting (XGBoost or LightGBM).
- Wrap preprocessing in a scikit-learn `Pipeline` so imputation and scaling are also fitted inside each cross-validation fold.

## Author
Aisha Hormuud University
