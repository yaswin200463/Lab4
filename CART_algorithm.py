from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree using CART (Gini Index)
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Pruning using GridSearchCV
param_grid = {
    'max_depth': [None, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4]
}
grid_search = GridSearchCV(DecisionTreeClassifier(criterion='gini', random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters after pruning
best_params = grid_search.best_params_
print("Best Parameters after Pruning:", best_params)

# Train the pruned tree
pruned_model = grid_search.best_estimator_
pruned_model.fit(X_train, y_train)

# Evaluate the pruned tree
y_pruned_pred = pruned_model.predict(X_test)
pruned_accuracy = accuracy_score(y_test, y_pruned_pred)
print(f"Pruned Accuracy: {pruned_accuracy}")

# Visualize the pruned decision tree
plt.figure(figsize=(12, 8))
plot_tree(pruned_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
