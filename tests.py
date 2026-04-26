from database_handler import prepare_dataset
from tree import DecisionTree

import csv
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def run_custom_tree(max_depth, min_samples_split, X_train, y_train, X_test, y_test, test_size, writer):
    dt = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
    dt.fit(X_train, y_train)

    tp = tn = fp = fn = 0

    n = len(y_test)
    right = 0
    for i in range (n):
        pred = dt.predict_sample(X_test[i], dt.root)
        if pred == y_test[i]:
            right+=1
        
        # calculating values in confusion matrix
        if y_test[i] == 1 and pred == 1:
            tp += 1
        elif y_test[i] == 0 and pred == 0:
            tn += 1
        elif y_test[i] == 0 and pred == 1:
            fp += 1
        elif y_test[i] == 1 and pred == 0:
            fn += 1

    cm = [[tn, fp], [fn, tp]]
    acc = right/n
    writer.writerow(['CustomTree', test_size, 'max_depth', max_depth, acc, tn, fp, fn, tp])

def run_tree(max_depth, min_samples_split, X_train, y_train, X_test, y_test, test_size, writer):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test)
    
    # calculate acc
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    writer.writerow(['SklearnTree', test_size, 'max_depth', max_depth, acc, tn, fp, fn, tp])

def run_forest(max_depth, n_estimators, random_state, X_train, y_train, X_test, y_test, test_size, writer):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test)
    
    # calculate acc
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    writer.writerow(['RandomForrest', test_size, 'max_depth', max_depth, acc, tn, fp, fn, tp])

def run_regression(X_train, y_train, X_test, y_test, test_size, writer):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test)

     # calculate acc
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    writer.writerow(['LogisticRegression', test_size, '-', '-', acc, tn, fp, fn, tp])

def run_mlp(hidden_layer_sizes, max_iter, random_state, X_train, y_train, X_test, y_test, test_size, writer):
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, max_iter = max_iter, random_state = random_state)
    model.fit(X_train, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test)

    # calculate acc
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()


    param = f"layers:{hidden_layer_sizes}_iters:{max_iter}"
    writer.writerow(['MLP', test_size, 'param', param, acc, tn, fp, fn, tp])

def run_tests(dataset_name = "personality_datasert.csv", output_file="wyniki_modeli.csv"):
    test_sizes = [0.2, 0.3, 0.5] #splits of dataset
    max_depths = [5, 10, 20] # max_depth for tree and forrest
    mlp_max_iters = [500, 1000] # iter for mlp
    mlp_hidden_layers = [(5,), (10, 5)] #layers for mlp

    headers = ['Model', 'Test_Size', 'Param_Name', 'Param_Value', 'Accuracy', 'TN', 'FP', 'FN', 'TP']
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        print("Started running tests")

        for ts in test_sizes:
            X_train, X_test, y_train, y_test = prepare_dataset(name=dataset_name, test_size=ts)
        
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #test regression
            run_regression(X_train_scaled, y_train, X_test_scaled, y_test, ts, writer)

            #test max_depth
            for depth in max_depths:
                run_custom_tree(depth, 2, X_train, y_train, X_test, y_test, ts, writer)

            for depth in max_depths:
                run_tree(depth, 2, X_train_scaled, y_train, X_test_scaled, y_test, ts, writer)

            for depth in max_depths:
                run_forest(depth, 10, 42, X_train, y_train, X_test, y_test, ts, writer)

            #iters and layers mlp
            for iters in mlp_max_iters:
                for layers in mlp_hidden_layers:
                    run_mlp(layers, iters, 42, X_train_scaled, y_train, X_test_scaled, y_test, ts, writer)

        print("Tests ended")