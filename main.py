from database_handler import prepare_dataset
from tree import DecisionTree
from tests import run_tests

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = prepare_dataset(name="personality_datasert.csv", test_size=0.3)

#scaled data for logistic regresiion and NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def tree(max_depth=20, min_samples_split=2):
    dt = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
    dt.fit(X_train, y_train)

    # for confusion matrix
    tp = tn = fp = fn = 0

    n = 50000 #how many times should the tests be computed
    if n > len(y_test):
        n = len(y_test)
    right = 0
    for i in range (n):
        print("SAMPLE ", i)
        pred = dt.predict_sample(X_test[i], dt.root)
        print("Prediction: ", pred)
        print("Real class: ", y_test[i])
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
    print("******************************************************************")
    print("Testing finished with ", n, "samples; results:")

    cm = [[tn, fp], [fn, tp]]
    print(f"Decision Tree Confusion Matrix:")
    print(cm)

    acc = right/n
    print(f"Decision Tree Accuracy: {acc:.4f}")
    #print("Tree:")
    # dt.print_tree()

def logistic_regression():
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test_scaled)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Logistic Regresion Confusion Matrix:")
    print(cm)
    
    # calculate acc
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression Accuracy: {acc:.4f}")

def random_forest(n_estimators=10, max_depth=10, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Random Forest Confusion Matrix:")
    print(cm)
    
    # calculate acc
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Randon Forest Accuracy: {acc:.4f}")

def mlp(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42):
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, max_iter = max_iter, random_state = random_state)
    model.fit(X_train, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test_scaled)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"MLP Confusion Matrix:")
    print(cm)
    
    # calculate acc
    acc = accuracy_score(y_test, y_pred)
    
    print(f"MLP Accuracy: {acc:.4f}")

def tree_sklearn(max_depth = 20, min_samples_split=2):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train_scaled, y_train)
    
    # test model on X test
    y_pred = model.predict(X_test_scaled)
    
    # calculate acc
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Sklearn Tree Confusion Matrix:")
    print(cm)
    
    print(f"Sklearn Tree Accuracy: {acc:.4f}")

# tree()
# logistic_regression()
# random_forest()
# mlp()
# tree_sklearn(10, 2)

run_tests()