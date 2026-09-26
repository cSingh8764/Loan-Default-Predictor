'''
Loan Default Predictor
main.py

Pipeline: clean data -> explore -> train -> evaluate on held-out test set -> predict new requests.

Run:  py main.py              (shows charts, then the interactive carousel)
      py main.py --no-plots   (skips the charts)
'''
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from carousel import Carousel

TRAIN_FILE = "credit_risk_train.csv"
TEST_FILE = "credit_risk_test.csv"
CLEANED_TRAIN_FILE = "credit_risk_train_clean.csv"
REQUEST_FILE = "loan_requests.csv"

TARGET = "loan_status"
AGE_LIMIT = 90

# Text columns. The model needs these turned into numbers (see build_model).
CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
NUMERIC = ["person_age", "person_income", "person_emp_length", "loan_amnt",
           "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
FEATURES = CATEGORICAL + NUMERIC

# Probability above which we predict "will default".
# Chosen with cross-validation on the TRAINING set only (F1-optimal was ~0.48, so 0.5 is kept).
THRESHOLD = 0.5


# ---------------------------------------------------------------- cleaning
def clean_training_data(df):
    '''Drops impossible ages. Missing values are kept: the model handles them natively.'''
    print(f"Initial number of rows: {len(df)}")
    missing = df.isna().sum()
    for col, n in missing[missing > 0].items():
        print(f"Column {col}: {n} values missing (kept; model handles missing values)")

    overage = df["person_age"] > AGE_LIMIT
    print(f"Number of records with age > {AGE_LIMIT}: {overage.sum()} (removed)")
    df = df[~overage]
    print(f"Remaining number of rows: {len(df)}\n")
    return df


# ---------------------------------------------------------------- charts
def age_histogram(df):
    bins = range(10, 101, 10)
    plt.figure(figsize=(12, 6))
    plt.hist(df.loc[df[TARGET] == 1, "person_age"], bins=bins, alpha=0.6,
             label="In Default", color="red", edgecolor="black")
    plt.hist(df.loc[df[TARGET] == 0, "person_age"], bins=bins, alpha=0.6,
             label="Not in Default", color="black", edgecolor="black")
    plt.xlabel("Age (in years)")
    plt.ylabel("No. of Borrowers")
    plt.title("Loan Distribution by Age")
    plt.legend()
    plt.tight_layout()
    plt.show()


def homeowner_pie(df):
    owners = df[df["person_home_ownership"] == "OWN"]
    sizes = [(owners[TARGET] == 1).sum(), (owners[TARGET] == 0).sum()]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=(0.1, 0), labels=["Defaulted", "Not Defaulted"],
            colors=["red", "green"], autopct="%1.1f%%", startangle=140)
    plt.title("Homeowners: Default vs. Not Default")
    plt.axis("equal")
    plt.show()


def default_rate_by_grade(df):
    '''The strongest single signal in the data: default rate climbs steeply with loan grade.'''
    rates = df.groupby("loan_grade")[TARGET].mean().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(rates.index, rates.values * 100, color="red", edgecolor="black")
    plt.xlabel("Loan grade (A = safest)")
    plt.ylabel("Default rate (%)")
    plt.title("Default Rate by Loan Grade")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- model
def build_model():
    '''
    Gradient-boosted trees: hundreds of small trees, each one correcting the errors of the
    ones before it. Handles missing values natively and needs no feature scaling
    (tree splits only compare values, so scale doesn't matter).
    The OrdinalEncoder turns text categories (e.g. "RENT", "OWN") into integer codes,
    and categorical_features tells the model to treat those codes as categories, not numbers.
    '''
    encoder = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ("passthrough", NUMERIC),
    )
    model = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CATEGORICAL))),  # first 4 columns after encoding
        random_state=42,
    )
    return make_pipeline(encoder, model)


def cross_validate(model, X, y):
    '''5-fold CV on training data only: an estimate of performance before touching the test set.'''
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold cross-validation ROC-AUC (train): {scores.mean():.3f} +/- {scores.std():.3f}")


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    # Baseline: always predict "no default". Any model must beat this to be useful.
    baseline_acc = (y_test == 0).mean()

    print("\n================ Evaluation on held-out test set ================")
    print(f"Test rows: {len(y_test)}   Default rate: {y_test.mean():.1%}")
    print(f"Baseline accuracy (always 'no default'): {baseline_acc:.3f}")
    print(f"Model accuracy:                          {accuracy_score(y_test, preds):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probs):.3f}   (1.0 = perfect ranking, 0.5 = random)")
    print(f"PR-AUC:   {average_precision_score(y_test, probs):.3f}   (baseline = {y_test.mean():.3f})")
    print("\nClassification Report (class 1 = default):")
    print(classification_report(y_test, preds, digits=3))
    print("Confusion Matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, preds))

    # Lowering the threshold catches more defaulters but rejects more good borrowers.
    print("\nThreshold trade-off (for the default class):")
    print("threshold  precision  recall")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
        p = (probs >= t).astype(int)
        print(f"   {t:.1f}      {precision_score(y_test, p):.3f}     {recall_score(y_test, p):.3f}")


def top_features(model, X_test, y_test):
    '''Permutation importance: shuffle one column, measure how much ROC-AUC drops.'''
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                    n_repeats=5, random_state=42, n_jobs=-1)
    ranked = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("\nMost important features (drop in ROC-AUC when shuffled):")
    for name, value in ranked.head(5).items():
        print(f"  {name:28s} {value:.3f}")


# ---------------------------------------------------------------- deployment
def deploy_predictor(model, file_path):
    requests = pd.read_csv(file_path)
    probs = model.predict_proba(requests[FEATURES])[:, 1]

    carousel = Carousel()
    for (_, row), prob in zip(requests.iterrows(), probs):
        record = row.to_dict()
        record["prob"] = prob
        record["prediction"] = int(prob >= THRESHOLD)
        carousel.add(record)

    print("\nPredicted Loan Status for Requests:")
    print([int(p >= THRESHOLD) for p in probs])

    input("\nPress Enter to view carousel interface...")

    while True:
        current = carousel.getCurrentData()
        print("\n--------------------------------------------------")
        print(f"Borrower: {current['borrower']}\n")
        print(f"Age: {current['person_age']}\n")
        print(f"Income: ${current['person_income']}\n")
        print(f"Home_ownership: {current['person_home_ownership']}\n")
        print(f"Employment: {current['person_emp_length']}\n")
        print(f"Loan intent: {current['loan_intent']}\n")
        print(f"Loan grade: {current['loan_grade']}\n")
        print(f"Amount: ${current['loan_amnt']}\n")
        print(f"Interest Rate: {current['loan_int_rate']}\n")
        print(f"Loan percent income: {current['loan_percent_income']}\n")
        default_flag = "Yes" if str(current['cb_person_default_on_file']).strip().upper() == "Y" else "No"
        print(f"Historical Defaults: {default_flag}\n")
        print(f"Credit History: {current['cb_person_cred_hist_length']} years\n")
        print("--------------------------------------------------")
        status_msg = "Will default" if current['prediction'] == 1 else "Will not default"
        recommendation = "Reject" if current['prediction'] == 1 else "Accept"
        print(f"Probability of default: {current['prob']:.1%}")
        print(f"Predicted loan_status: {status_msg}")
        print(f"Recommend: {recommendation}")
        print("--------------------------------------------------")

        choice = input("Enter 1 for next, 2 for previous, 0 to quit: ")
        if choice == '1':
            carousel.moveNext()
        elif choice == '2':
            carousel.movePrevious()
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")


# ---------------------------------------------------------------- main
def main():
    show_plots = "--no-plots" not in sys.argv

    train = clean_training_data(pd.read_csv(TRAIN_FILE))
    train.to_csv(CLEANED_TRAIN_FILE, index=False)
    test = pd.read_csv(TEST_FILE)

    if show_plots:
        age_histogram(train)
        homeowner_pie(train)
        default_rate_by_grade(train)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = build_model()
    cross_validate(model, X_train, y_train)
    model.fit(X_train, y_train)          # train on training data only
    evaluate(model, X_test, y_test)      # test set is used exactly once, here
    top_features(model, X_test, y_test)
    deploy_predictor(model, REQUEST_FILE)


if __name__ == "__main__":
    main()
