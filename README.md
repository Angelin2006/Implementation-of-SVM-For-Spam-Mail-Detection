# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.
2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head(). 
3.Split the dataset using train_test_split. 
4.Calculate Y_Pred and accuracy. 
5.Print all the outputs. 
6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: C. Angelin Jebisha
RegisterNumber: 212224060019
*/


# Import necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data

df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the first two columns and rename them

df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Convert labels(ham = 0, spam = 1) :

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove any rows with missing values

df = df.dropna()

# Display basic info

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())

# Split data into features (X) and target (y)

X = df['message']
y = df['label']

# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train SVM model

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vectorized, y_train)

# Make predictions

y_pred = svm_model.predict(X_test_vectorized)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Test with some example messages

test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won a FREE cruise to the Andaman! Call now to claim your prize!",
    "Can you pick up some milk on your way home?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately."
]

print("\n" + "="*50)
print("TESTING WITH EXAMPLE MESSAGES:")
print("="*50)

for msg in test_messages:
    msg_vectorized = vectorizer.transform([msg])
    prediction = svm_model.predict(msg_vectorized)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: {msg[:50]}... -> {result}")

```

## Output:

DATASET SHAPE :

<img width="782" height="262" alt="image" src="https://github.com/user-attachments/assets/862f3409-91ee-4090-8827-e7e94dfadd7e" />

CLASS DISTRIBUTION :

<img width="421" height="196" alt="image" src="https://github.com/user-attachments/assets/f36d2780-aaa7-430b-8573-8870290de660" />

CLASSIFICATION REPORT :

<img width="682" height="251" alt="image" src="https://github.com/user-attachments/assets/1e51cd4d-5aed-456c-8636-1c0101418125" />

SAMPLE :

<img width="878" height="178" alt="image" src="https://github.com/user-attachments/assets/e3778049-954d-4d47-a753-7d65493dc072" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
