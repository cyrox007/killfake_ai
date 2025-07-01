import os

import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Абсолютный путь до проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(filepath='ds/fake_news.csv'):
    """ Загрузка данных из датасета """
    data_frame = pd.read_csv(os.path.join(BASE_DIR, filepath))
    print(f"Данные загружены. Количество записей: {len(data_frame)}")
    return data_frame

def preprocess_data(df: DataFrame):
    """ предобработка данных датасета """
    df.dropna(inplace=True)

    X = df['text']
    Y = df['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def vectorizer_data(X_train, X_test):
    """Векторизация текста с помощью TF-IDF"""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print("TF-IDF векторизация завершена.")
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def train_model(X_train_tfidf, y_train):
    """Обучение модели PassiveAggressiveClassifier"""
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    pac.fit(X_train_tfidf, y_train)

    print("Модель обучена.")
    return pac

def evaluate_model(model, X_test_tfidf, y_test):
    """Оценка качества модели"""
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nТочность модели: {round(accuracy * 100, 2)}%\n')

    # Отчет по классификации
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Предсказанные метки")
    plt.ylabel("Истинные метки")
    plt.title("Матрица ошибок")
    plt.show()

def visualize_top_features(model, vectorizer, top_n=10):
    """Визуализация наиболее важных слов"""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    # Сортировка самых важных слов
    top = np.argsort(coefs)[-top_n:]
    bottom = np.argsort(coefs)[:top_n]

    print("\nНаиболее характерные слова для 'FAKE':")
    print(feature_names[bottom])

    print("\nНаиболее характерные слова для 'REAL':")
    print(feature_names[top])

    # График
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), coefs[top], align='center', color='green', label='REAL')
    plt.barh(range(top_n), coefs[bottom], align='center', color='red', label='FAKE')
    plt.yticks(range(top_n), feature_names[top])
    plt.title("Наиболее информативные слова")
    plt.legend()
    plt.show()


def run():
    df = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(df)
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorizer_data(x_train, x_test)
    model = train_model(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test)
    visualize_top_features(model, tfidf_vectorizer)


if __name__ == "__main__":
    run()