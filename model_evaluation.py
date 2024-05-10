import streamlit as st
from sklearn.metrics import accuracy_score
from train_models import X_test, y_test, classifier, kmeans, svm_classifier


# Вычисление производительности моделей на тестовом наборе данных
def evaluate_models():
    # Оценка производительности модели решающего дерева
    y_pred = classifier.predict(X_test)
    decision_tree_accuracy = accuracy_score(y_test, y_pred)

    # Оценка производительности модели KMeans
    y_pred = kmeans.predict(X_test)
    kmeans_accuracy = accuracy_score(y_test, y_pred)

    # Оценка производительности модели SVM
    y_pred = svm_classifier.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred)

    return decision_tree_accuracy, kmeans_accuracy, svm_accuracy

# Оценка производительности моделей
decision_tree_accuracy, kmeans_accuracy, svm_accuracy = evaluate_models()

# Веб-приложение для визуализации результатов
st.title("Model Evaluation")

st.write("Accuracy of Decision Tree Classifier:", decision_tree_accuracy)
st.write("Accuracy of KMeans:", kmeans_accuracy)
st.write("Accuracy of Support Vector Machine:", svm_accuracy)
