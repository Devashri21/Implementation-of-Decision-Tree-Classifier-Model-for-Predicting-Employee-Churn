# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries (pandas, numpy, sklearn).
2.Load the dataset Employee.csv into a pandas DataFrame.
3.Display the first few rows and check for missing values.
4.Encode the categorical variables using LabelEncoder.
5.Split the dataset into features (X) and target (y).
6.Split the data into training and testing sets using train_test_split().
7.Create an instance of DecisionTreeClassifier.
8.Train the model using the training data.
9.Predict the churn status on the test data.
10.Evaluate model performance using accuracy_score and classification_report.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Employee.csv")

le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

X = data.drop(['left'], axis=1)
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy of the model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```

## Output:
<img width="828" height="279" alt="image" src="https://github.com/user-attachments/assets/55fbba0d-c595-40e2-b409-488f1f0dee10" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
