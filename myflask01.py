# add label encoding for tables

from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np  # Import numpy for random seed
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import io
from io import BytesIO

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    global clean_decision_table_m1
    global clean_decision_table_m2
    cv_splits_m1 = 5
    
    if request.method == 'POST':
        
        if 'cv_splits_m1' in request.form:
            cv_splits_m1 = int(request.form['cv_splits_m1'])
            
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return redirect(request.url)

        # Check if the file is a CSV file
        if file and file.filename.endswith('.csv'):
            # Save the uploaded file
            file.save('uploads/' + file.filename)

            # Read CSV into a pandas DataFrame
            decision_table = pd.read_csv('uploads/' + file.filename)
            
            # Get the number of rows and columns
            num_rows, num_columns = decision_table.shape
            
            # Get the list of all columns except the last one
            columns_to_check = decision_table.columns[:-1]
            # Get the last column
            last_column = decision_table.columns[-1]
            
            # Check for duplicated rows
            duplicated_rows = decision_table[decision_table.duplicated(columns_to_check, keep=False)]
            #Get the number of rows in clean_decision_table_m1
            num_duplicated_rows = duplicated_rows.shape[0]
            
            # Initialize a list to store inconsistent rows
            inconsistent_rows_f = []
            
            for _, group in duplicated_rows.groupby(columns_to_check.tolist()):
                unique_decisions = group[last_column].unique()

                if len(unique_decisions) > 1:
                    # If there are different values in last_column, it's inconsistent
                    inconsistent_rows_f.append(group)
            
            if inconsistent_rows_f:    
            # Concatenate the inconsistent rows into a DataFrame
                inconsistent_rows = pd.concat(inconsistent_rows_f)
            
            if not inconsistent_rows.empty:
                inconsistent_exist = "Yes"

                # Remove inconsistent rows from the original DataFrame - method 1
                clean_decision_table_m1 = decision_table.drop(inconsistent_rows.index)
                # Remove duplicate rows, keeping only the first occurrence
                clean_decision_table_m1 = clean_decision_table_m1.drop_duplicates(columns_to_check)
                #Get the number of rows in clean_decision_table_m1
                num_rows_clean_m1 = clean_decision_table_m1.shape[0]

                
                # Resolve conflicts by keeping one row for each unique combination of attributes - method 2
                clean_decision_table_m2 = pd.DataFrame()
    
                for _, group in decision_table.groupby(columns_to_check.tolist()):
                    unique_decisions = group[last_column].unique()
                    
                    if len(unique_decisions) == 1:
                        # If there's only one unique decision, choose randomly from the group
                        np.random.seed()  # Reset the random seed
                        chosen_row = group.sample(n=1)
                    else:
                        # If there are multiple decisions, choose randomly among the most frequent decisions
                        most_frequent_decision_count = group[last_column].value_counts().max()
                        most_frequent_decisions = group[last_column].value_counts()[group[last_column].value_counts() == most_frequent_decision_count].index
                        chosen_decision = np.random.choice(most_frequent_decisions)
                        chosen_row = group[group[last_column] == chosen_decision].sample(n=1)
                        
                    clean_decision_table_m2 = pd.concat([clean_decision_table_m2, chosen_row])
                    
                    #Get the number of rows in clean_decision_table_m2
                    num_rows_clean_m2 = clean_decision_table_m2.shape[0]
            else:    
                inconsistent_exist = "No"
           
            # METHOD 1 - TRAIN-TEST
            
            # Features (X) are columns_to_check, and the label (y) is last_column
            X_m1 = clean_decision_table_m1[columns_to_check]
            y_m1 = clean_decision_table_m1[last_column]
            
            # Split the data into train and test sets for method 1
            X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(X_m1, y_m1, test_size=0.25)

            # Train a decision tree classifier for method 1
            clf_m1 = DecisionTreeClassifier()
            clf_m1.fit(X_train_m1, y_train_m1)

            # Get information about the decision tree for train-test split
            depth_m1_train_test = clf_m1.get_depth()
            nodes_m1_train_test = clf_m1.get_n_leaves()

            # Evaluate the classifier on the test set for method 1
            accuracy_m1_tt = round(clf_m1.score(X_test_m1, y_test_m1), 2)
                
            # METHOD 1 - K-FOLD
            
            average_accuracy_m1_kf = 0.0
            
            if num_rows_clean_m1 < 10 and cv_splits_m1==10:
                cv_splits_m1=5
            elif num_rows_clean_m1 < 5 and (cv_splits_m1==5 or cv_splits_m1==10):    
                cv_splits_m1 = min(5, len(X_m1))
            
            if cv_splits_m1 >= 5:
                cv_m1 = KFold(n_splits=cv_splits_m1, shuffle=True)

                # Create a decision tree classifier for method 1
                clf_m1 = DecisionTreeClassifier()

                # Initialize lists to store results for each split
                depths_m1_splits = []
                nodes_m1_splits = []
                accuracies_m1_splits = []

                # Perform cross-validation for each split
                for i, (train_indices, test_indices) in enumerate(cv_m1.split(X_m1, y_m1), start=1):
                    # Fit the classifier on the training data for the current split
                    clf_m1.fit(X_m1.iloc[train_indices], y_m1.iloc[train_indices])

                    # Get information about the decision tree for the current split
                    depth_m1_split = clf_m1.get_depth()
                    nodes_m1_split = clf_m1.get_n_leaves()

                    # Evaluate the classifier on the test set for the current split
                    accuracy_m1_split = cross_val_score(clf_m1, X_m1, y_m1, cv=cv_m1, scoring='accuracy')[i-1]

                    # Append results to the lists
                    depths_m1_splits.append(depth_m1_split)
                    nodes_m1_splits.append(nodes_m1_split)
                    accuracies_m1_splits.append(accuracy_m1_split)
                
                # Get the average accuracy across all splits
                average_depth_m1_kf = round(sum(depths_m1_splits) / cv_splits_m1, 2)
                average_nodes_m1_kf = round(sum(nodes_m1_splits) / cv_splits_m1, 2)
                average_accuracy_m1_kf = round(sum(accuracies_m1_splits) / cv_splits_m1, 2)
                

            else:
                depths_m1_splits = [0]*cv_splits_m1
                nodes_m1_splits = [0]*cv_splits_m1
                accuracies_m1_splits = [0]*cv_splits_m1
                
            std_accuracy_m1_kf = round(np.std(accuracies_m1_splits), 2)
                
            # METHOD 2 - TRAIN-TEST
            
            # Features (X) are columns_to_check, and the label (y) is last_column
            X_m2 = clean_decision_table_m2[columns_to_check]
            y_m2 = clean_decision_table_m2[last_column]
            
            # Split the data into train and test sets for METHOD 2
            X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(X_m2, y_m2, test_size=0.25)

            # Train a decision tree classifier for METHOD 2
            clf_m2 = DecisionTreeClassifier()
            clf_m2.fit(X_train_m2, y_train_m2)

            # Get information about the decision tree for train-test split
            depth_m2_train_test = clf_m2.get_depth()
            nodes_m2_train_test = clf_m2.get_n_leaves()

            # Evaluate the classifier on the test set for METHOD 2
            accuracy_m2_tt = round(clf_m2.score(X_test_m2, y_test_m2), 2)
                
            # METHOD 2 - K-FOLD
            
            average_accuracy_m2_kf = 0.0
            
            if num_rows_clean_m2 < 10 and cv_splits_m1==10:
                cv_splits_m1=5
            elif num_rows_clean_m2 < 5 and (cv_splits_m1==5 or cv_splits_m1==10):    
                cv_splits_m1 = min(5, len(X_m2))

            if cv_splits_m1 >= 5:
                cv_m2 = KFold(n_splits=cv_splits_m1, shuffle=True)

                # Create a decision tree classifier for METHOD 2
                clf_m2 = DecisionTreeClassifier()

                # Initialize lists to store results for each split
                depths_m2_splits = []
                nodes_m2_splits = []
                accuracies_m2_splits = []

                # Perform cross-validation for each split
                for i, (train_indices, test_indices) in enumerate(cv_m2.split(X_m2, y_m2), start=1):
                    # Fit the classifier on the training data for the current split
                    clf_m2.fit(X_m2.iloc[train_indices], y_m2.iloc[train_indices])

                    # Get information about the decision tree for the current split
                    depth_m2_split = clf_m2.get_depth()
                    nodes_m2_split = clf_m2.get_n_leaves()

                    # Evaluate the classifier on the test set for the current split
                    accuracy_m2_split = cross_val_score(clf_m2, X_m2, y_m2, cv=cv_m2, scoring='accuracy')[i-1]

                    # Append results to the lists
                    depths_m2_splits.append(depth_m2_split)
                    nodes_m2_splits.append(nodes_m2_split)
                    accuracies_m2_splits.append(accuracy_m2_split)
        
                # Get the average accuracy across all splits
                average_depth_m2_kf = round(sum(depths_m2_splits) / cv_splits_m1, 2)
                average_nodes_m2_kf = round(sum(nodes_m2_splits) / cv_splits_m1, 2)
                average_accuracy_m2_kf = round(sum(accuracies_m2_splits) / cv_splits_m1, 2)
                
            else:
                depths_m2_splits = [0]*cv_splits_m1
                nodes_m2_splits = [0]*cv_splits_m1
                accuracies_m2_splits = [0]*cv_splits_m1
            
            std_accuracy_m2_kf = round(np.std(accuracies_m2_splits), 2)    
            # Pass the data to the template
            table_data = decision_table.head(5)
            if not table_data.empty:
                return render_template('index.html', num_duplicated_rows=num_duplicated_rows, num_rows=num_rows, num_columns=num_columns, table=table_data, inconsistent_rows=len(inconsistent_rows), inconsistent_exist=inconsistent_exist, depth_m1_train_test=depth_m1_train_test, nodes_m1_train_test=nodes_m1_train_test, accuracy_m1_tt=accuracy_m1_tt, average_accuracy_m1_kf=average_accuracy_m1_kf, depths_m1_splits=depths_m1_splits, nodes_m1_splits=nodes_m1_splits, accuracies_m1_splits=accuracies_m1_splits, average_depth_m1_kf=average_depth_m1_kf, average_nodes_m1_kf=average_nodes_m1_kf, depth_m2_train_test=depth_m2_train_test, nodes_m2_train_test=nodes_m2_train_test, accuracy_m2_tt=accuracy_m2_tt, average_accuracy_m2_kf=average_accuracy_m2_kf, depths_m2_splits=depths_m2_splits, nodes_m2_splits=nodes_m2_splits, accuracies_m2_splits=accuracies_m2_splits, average_depth_m2_kf=average_depth_m2_kf,  average_nodes_m2_kf=average_nodes_m2_kf, num_rows_clean_m1=num_rows_clean_m1, num_rows_clean_m2=num_rows_clean_m2, cv_splits_m1=cv_splits_m1, std_accuracy_m1_kf=std_accuracy_m1_kf, std_accuracy_m2_kf=std_accuracy_m2_kf)

    return render_template('index.html')

@app.route('/download_csv_m1')
def download_csv_m1():
    global clean_decision_table_m1
    csv_io_m1 = BytesIO()
    clean_decision_table_m1.to_csv(csv_io_m1, index=False, encoding='utf-8')
    csv_io_m1.seek(0)
    return send_file(csv_io_m1, mimetype='text/csv', as_attachment=True, download_name='data_m1.csv')

@app.route('/download_csv_m2')
def download_csv_m2():
    global clean_decision_table_m2
    csv_io_m2 = BytesIO()
    clean_decision_table_m2.to_csv(csv_io_m2, index=False, encoding='utf-8')
    csv_io_m2.seek(0)
    return send_file(csv_io_m2, mimetype='text/csv', as_attachment=True, download_name='data_m2.csv')

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
