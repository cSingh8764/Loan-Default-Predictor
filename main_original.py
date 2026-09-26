'''
Loan Default Predictor
main.py
'''
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from carousel import Carousel   

#reads the csv file and returns the header and data rows of the file
def read_csv_file(file):
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    header = lines[0].strip().split(',')
    data_rows = [line.strip().split(',') for line in lines[1:] if line.strip()]
    return header, data_rows

#removes the rows with missing values and couns how many values are missing in each column
#returns cleaned data, missing value counts, and the orignal number of rows
def Removing_missing_Values(header, data_rows):
    column_num = len(header)
    missing_count = [0] * column_num
    clean_rows = []
    initial_row_count = len(data_rows)

    for row in data_rows:
        if len(row) != column_num:
            continue

        is_missing = False
        for i in range(column_num):
            if row[i].strip() == '':
                missing_count[i] += 1
                is_missing = True
        if not is_missing:
            clean_rows.append(row)

    return clean_rows, missing_count, initial_row_count

def remove_overage_applicants(header, data_rows):
    
    age_limit = 90
    
    age_index = header.index("person_age")
    
    final_rows = []
    overage_count = 0
    
    for row in data_rows:
        try:
            age = int(row[age_index])
            if age <= age_limit:
                final_rows.append(row)
            else:
                overage_count += 1
        except ValueError:
            overage_count += 1

    return final_rows, overage_count

#writes the clean data into a new CSV file
#the orignal file can be change if we use the same path as the orignal file  
def write_cleaned_data(file_path, header, data_rows):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in data_rows:
            f.write(','.join(row) + '\n')

#displays a histogram of age distrubution for defaulted and non-defaulted loans
def bar_graph(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    age_index = header.index('person_age')
    status_index = header.index('loan_status')

    default_ages = []
    non_default_ages = []

    for line in lines[1:]:
        row = line.strip().split(',')

        age = int(row[age_index])
        status = row[status_index].strip().lower()

        if status == '1':
            default_ages.append(age)
        else:
            non_default_ages.append(age)

    bins = range(10, 101, 10)  # bins like 10-20, 20-30, ..., 90-100

    plt.figure(figsize=(12, 6))
    plt.hist(default_ages, bins=bins, alpha=0.6, label='In Default', color='red', edgecolor='black')
    plt.hist(non_default_ages, bins=bins, alpha=0.6, label='Not in Default', color='black', edgecolor='black')

    plt.xlabel("Age (in years)")
    plt.ylabel("No. of Borrowers")
    plt.title("Loan Distribution by Age (Histogram)")
    plt.legend()
    plt.tight_layout()
    plt.show()

#displays a pie chart of default status among homeowners
def pie_chart(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')

    # Get column indexes
    home_index = header.index("person_home_ownership")
    status_index = header.index("loan_status")

    # Counters
    defaulted = 0
    not_defaulted = 0

    for line in lines[1:]:
        row = line.strip().split(',')
        if len(row) <= max(home_index, status_index):
            continue  # skip incomplete rows

        home_ownership = row[home_index].strip().upper()
        loan_status = row[status_index].strip()

        if home_ownership == "OWN":
            if loan_status == "1":
                defaulted += 1
            elif loan_status == "0":
                not_defaulted += 1

    total = defaulted + not_defaulted
    if total == 0:
        print("No homeowners found in the dataset.")
        return

    # Pie chart
    labels = ['Defaulted', 'Not Defaulted']
    sizes = [defaulted, not_defaulted]
    colors = ['red', 'green']
    explode = (0.1, 0)  # explode the 'Defaulted' slice

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Homeowners: Default vs. Not Default')
    plt.axis('equal')  # Make pie chart a circle
    plt.show()

#counts number of dafualt and non-defaults in the datasets
def count_default_status(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split(',')
    status_index = header.index("loan_status")
    
    defaulted = 0
    not_defaulted = 0
    
    for line in lines[1:]:
        row = line.strip().split(',')

        loan_status = row[status_index].strip()
        if loan_status == "1":
            defaulted += 1
        elif loan_status == "0":
            not_defaulted += 1

    return defaulted, not_defaulted

#scales income and loan amont features using standarddcaler
def scale_features(header, data_rows):
    income_index = header.index("person_income")
    loan_index = header.index("loan_amnt")

    features_to_scale = []
    for row in data_rows:
        income = float(row[income_index])
        loan = float(row[loan_index])
        features_to_scale.append([loan, income])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_scale)

    for i in range(len(data_rows)):
        data_rows[i][loan_index] = str(scaled_features[i][0])
        data_rows[i][income_index] = str(scaled_features[i][1])

    return data_rows, scaler

#evaluates model performance with a diffrent file
def evaluate_model(clf, scaler, file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    
    income_index = header.index("person_income")
    loan_index = header.index("loan_amnt")
    credit_hist_index = header.index("cb_person_cred_hist_length")
    status_index = header.index("loan_status")
    
    x_test = []
    y_test = []
    unscaled_data = []
    credit_history = []
    
    for line in lines[1:]:
        row = line.strip().split(',')
        
        scaled_loan = float(row[loan_index])
        scaled_income = float(row[income_index])
        unscaled_data.append([scaled_loan, scaled_income])
        credit_hist = int(row[credit_hist_index])
        
        status = int(row[status_index])

        credit_history.append(credit_hist)
        y_test.append(status)
    scaled_data = scaler.fit_transform(unscaled_data)
    
    for i in range(len(scaled_data)):
        x_test.append([float(scaled_data[i][0]), float(scaled_data[i][1]), credit_history[i]])
    
    y_pred = clf.predict(x_test)
    
    print("Model Evaluation on Scaled Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

#Trans the decision tree model on the scaled traning database  
def train_model(data_rows, header):
    income_index = header.index("person_income")
    loan_index = header.index("loan_amnt")
    credit_index = header.index("cb_person_cred_hist_length")
    status_index = header.index("loan_status")

    x_train = []
    y_train = []
    
    for row in data_rows:
        loan = float(row[loan_index])
        income = float(row[income_index])
        credit = int(row[credit_index])
        status = int(row[status_index])
        
        x_train.append([loan, income, credit])
        y_train.append(status)
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(x_train, y_train)
    
    return clf

# uses the trained model to make predections on loan requests file
# adds the predicted enteries into a carousel and creates and interactive interface
def deploy_predictor(clf, scaler, file_path):
    header, data_rows = read_csv_file(file_path)

    income_index = header.index("person_income")
    loan_index = header.index("loan_amnt")
    credit_index = header.index("cb_person_cred_hist_length")

    carousel = Carousel()
    predictions = []

    for row in data_rows:
        loan = float(row[loan_index])
        income = float(row[income_index])
        credit = int(row[credit_index])

        scaled_loan, scaled_income = scaler.transform([[loan, income]])[0]
        prediction = int(clf.predict([[scaled_loan, scaled_income, credit]])[0])
        predictions.append(prediction)

        row_data = {header[i]: row[i] for i in range(len(header))}
        row_data["prediction"] = prediction

        carousel.add(row_data)

    print("\nPredicted Loan Status for Requests:")
    print(predictions)

    input("\nPress Enter to view carousel interface...")

    # Text-based interactive carousel to view one record at a time
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
        default_flag = "Yes" if current['cb_person_default_on_file'].strip().upper() == "Y" else "No"
        print(f"Historical Defaults: {default_flag}\n")
        print(f"Credit History: {current['cb_person_cred_hist_length']} years\n")
        print("--------------------------------------------------")
        status_msg = "Will default" if current['prediction'] == 1 else "Will not default"
        recommendation = "Reject" if current['prediction'] == 1 else "Accept"
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
            
# Main program execution: cleans, processes, trains, evaluates and deploys the model
def main():
    train_file = "credit_risk_train.csv"
    test_file = "credit_risk_test.csv"
    cleaned_train_file = "credit_risk_train_clean.csv"
    request_file = "loan_requests.csv"

    header, data_rows = read_csv_file(train_file)
    cleaned_rows, missing_counts, initial_row_count = Removing_missing_Values(header, data_rows)
    final_rows, overage_count = remove_overage_applicants(header, cleaned_rows)
    write_cleaned_data(cleaned_train_file, header, final_rows)

    print(f"Initial number of rows: {initial_row_count}")
    for i in range(len(header)):
        if missing_counts[i] > 0:
            print(f"Column {header[i]}: {missing_counts[i]} values missing")
    print()
    print(f"Number of records with age > 90: {overage_count}")
    print(f"Remaining number of rows: {len(final_rows)}")
    
    bar_graph(cleaned_train_file)
    pie_chart(cleaned_train_file)
    

    final_rows, scaler = scale_features(header, final_rows)
    clf = train_model(final_rows, header)
    evaluate_model(clf, scaler, test_file)
    deploy_predictor(clf, scaler, request_file)

    
if __name__ == "__main__":
    main()