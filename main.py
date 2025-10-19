# main.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# 1. Load the Dataset
# The Iris dataset is a classic dataset in machine learning and is included in scikit-learn.
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target # Labels (0: setosa, 1: versicolor, 2: virginica)
feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset loaded successfully.")
print(f"Features: {feature_names}")
print(f"Targets: {target_names}")
print("-" * 30)

# 2. Preprocess the Data
# For the Iris dataset, there are no missing values, and labels are already encoded numerically.
# This step is often more involved with real-world data.
print("Preprocessing:")
# Check for missing values (for demonstration)
df = pd.DataFrame(X, columns=feature_names)
print(f"Any missing values? {df.isnull().sum().any()}")
print("-" * 30)

# 3. Split Data into Training and Testing Sets
# We split the data to train the model on one subset and evaluate it on another, unseen subset.
# test_size=0.3 means 30% of the data is for testing, 70% for training.
# random_state ensures the split is the same every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("-" * 30)

# 4. Train a Decision Tree Classifier
# We create an instance of the DecisionTreeClassifier and train it using the .fit() method.
model = DecisionTreeClassifier(random_state=42)
print("Training the Decision Tree model...")
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)

# 5. Make Predictions on the Test Data
y_pred = model.predict(X_test)

# 6. Evaluate the Model
# We compare the predicted labels (y_pred) with the true labels (y_test).
accuracy = accuracy_score(y_test, y_pred)

# For multi-class classification, precision and recall require an averaging method.
# 'weighted' calculates metrics for each label and finds their average, weighted by support.
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))