from database_handler import prepare_dataset

X_train, X_test, y_train, y_test = prepare_dataset(name="personality_datasert.csv", test_size=0.3)