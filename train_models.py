from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pandas as pd
from preprocessing import preprocess_data

# Загрузка данных
file_path = "data/dataset.csv"
df = pd.read_csv(file_path)

# Предварительная обработка данных
preprocessed_df = preprocess_data(df)

# Разделение данных на признаки и целевую переменную
X = preprocessed_df.drop(columns=["Indicate your gender"])  # Исключаем столбец "Indicate your gender"
y = preprocessed_df["Indicate your gender"]  # Используем столбец "Indicate your gender" в качестве целевой переменной

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели с учителем (например, решающее дерево)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree Classifier:", accuracy)

# Обучение модели без учителя (например, KMeans)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of KMeans:", accuracy)

# Обучение модели машинного обучения на ваш выбор (например, SVM)
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Support Vector Machine:", accuracy)