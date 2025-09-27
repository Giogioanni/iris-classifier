# Giogioanni Morales 9/27/2025 last revision

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def main():
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    # Train and predict
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.2f}")

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=iris.target_names
    )
    plt.title("Iris Classifier Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()