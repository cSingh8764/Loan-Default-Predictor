# Loan Default Predictor

Predicts whether a loan applicant will default, using a gradient-boosted tree model trained on ~24k credit records, and lets you browse predictions for new applicants through a circular doubly-linked list interface.

Started as a CMPUT 175 (University of Alberta) assignment; the modelling and evaluation were later rebuilt.

## Results (held-out test set, 8,146 loans, 22.3% default rate)

| Model | Accuracy | ROC-AUC | Default recall | Default precision |
|---|---|---|---|---|
| Always predict "no default" (baseline) | 0.777 | 0.500 | 0.00 | – |
| v1: decision tree, 3 features | 0.73 | – | 0.48 | 0.41 |
| **v2: gradient boosting, 11 features** | **0.935** | **0.937** | **0.72** | **0.98** |

5-fold cross-validation ROC-AUC on the training set: 0.945 ± 0.003.

## What was wrong with v1, and what changed

1. **Worse than the baseline.** v1 scored 73% accuracy; predicting "no default" for everyone scores 78%. Accuracy hides this on imbalanced data, so v2 reports ROC-AUC, PR-AUC and per-class precision/recall against a baseline.
2. **Scaler re-fitted on the test set.** v1 called `fit_transform` on test data, so the test features were scaled with different statistics than the ones the tree was trained on. Fixing this alone raised accuracy from 0.73 to 0.79.
3. **Only 3 of 11 features used.** Loan grade, loan-to-income ratio, interest rate and home ownership are the strongest predictors and were ignored.
4. **Unlimited tree depth → overfitting.** Replaced with gradient boosting (`HistGradientBoostingClassifier`), which also handles missing values natively, so rows with missing interest rate or employment length are no longer discarded.

## Trade-off: missing defaulters vs rejecting good borrowers

The model outputs a probability; the decision threshold sets the trade-off. At 0.5 it misses ~28% of defaulters but is almost never wrong when it flags one. Lowering the threshold to 0.2 catches 81% of defaulters, at the cost of wrongly flagging more good applicants. The right threshold depends on what a missed default costs the lender compared with a rejected good customer; the program prints this table.

## Most important features

Measured with permutation importance (drop in ROC-AUC when a column is shuffled): loan-to-income ratio, home ownership, loan grade, income, loan intent.

## Run

```
py -m pip install -r requirements.txt
py main.py              # charts + evaluation + interactive carousel
py main.py --no-plots   # skip charts
```

## Files

```
main.py                 # cleaning, charts, training, evaluation, prediction interface
carousel.py             # circular doubly-linked list (course-provided scaffold)
credit_risk_train.csv   # training data
credit_risk_test.csv    # held-out test data
loan_requests.csv       # new applicants to predict
main_original.py        # v1, kept for comparison
```
