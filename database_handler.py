import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset(name="personality_datasert.csv", test_size=0.3):
    X, y = load_dataset(name=name)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    print(f"{name} dataset")

    print(f"Training set with {X_train.shape[0]} samples:")
    print(X_train)
    print(y_train)
    print(f"\nTest set with {X_test.shape[0]} samples:")
    print(X_test)
    print(y_test)

    return X_train, X_test, y_train, y_test

def load_dataset(name="personality_datasert.csv"):
    #1. pobierz dataset jako pandas
    df = pd.read_csv(name)
    #2. przerób yes/no na 0/1
    df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

    #3. Przerób Extravert/Introvert na 0/1
    df['Personality'] = df['Personality'].map({'Introvert': 1, 'Extrovert': 0})
    
    #4. Zamień dane na inty (dane są tylko w intach)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    X = df.iloc[:, :7].astype(int).values
    y = df.iloc[:, 7].astype(int).values
    #5. Zwróć gotowy dataset
    print(X)
    return X, y