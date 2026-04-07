from database_handler import prepare_dataset
from tree import DecisionTree

X_train, X_test, y_train, y_test = prepare_dataset(name="personality_datasert.csv", test_size=0.3)

dt = DecisionTree(max_depth=20, min_samples_split=2)
dt.fit(X_train, y_train)

n = 50 #how many times should the tests be computed
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
print("******************************************************************")
print("Testing finished with ", n, "samples; results:")
acc = right/n
print("Acc: ", acc)
#print("Tree:")
# dt.print_tree()