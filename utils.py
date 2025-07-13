"""
В этом файле происходит загружка и обработка данных, а также построение и обучение нейронной сети
Функциональность:
- Загрузка и кэширование признаков и меток пола из аудиофайлов
- Разделение выборки на тренировочную, валидационную и тестовую
- Построение многослойной нейронной сети с использованием Dropout для регуляризации

Архитектура модели подобрана экспериментально и сбалансирована по сложности и производительности.  
Моя цель — добиться максимальной точности без переобучения.
"""

import pandas as pd
import numpy as np
import os
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Сопоставление меток пола с числами: 1 — мужчина, 0 — женщина
gender_map = {
    "male": 1,
    "female": 0
}

def load_data(feature_vector_size=128):
    """
    Загружает аудиофичи и метки пола.
    Если данные уже сохранены в файлах .npy, загружает их, чтобы не пересчитывать заново.
    """
    # Проверяю, что папка results существует, если нет — создаю
    if not os.path.isdir("results"):
        os.mkdir("results")

    # Если файлы с признаками и метками уже есть — загружаю их
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        features = np.load("results/features.npy")
        targets = np.load("results/labels.npy")
        return features, targets

    # Иначе читаю CSV с информацией о файлах и поле 'gender'
    data_df = pd.read_csv("balanced-all.csv")

    total_samples = len(data_df)
    males_count = len(data_df[data_df['gender'] == 'male'])
    females_count = len(data_df[data_df['gender'] == 'female'])

    print(f"Всего примеров: {total_samples}")
    print(f"Мужчин: {males_count}")
    print(f"Женщин: {females_count}")

    # Инициализация массивов для фич и меток
    features = np.zeros((total_samples, feature_vector_size))
    targets = np.zeros((total_samples, 1))

    # Загружаю по одному файлу с признаками из .npy и соответствующую метку
    for idx, (fname, gender) in tqdm.tqdm(enumerate(zip(data_df['filename'], data_df['gender'])), total=total_samples, desc="Загрузка данных"):
        feat = np.load(fname)
        features[idx] = feat
        targets[idx] = gender_map[gender]

    # Сохраняю для повторного использования
    np.save("results/features", features)
    np.save("results/labels", targets)

    return features, targets

def split_data(features, targets, test_ratio=0.1, valid_ratio=0.1):
    """
    Делит данные на обучающую, валидационную и тестовую выборки.
    Фиксирую random_state для воспроизводимости.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, targets, test_size=test_ratio, random_state=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=valid_ratio, random_state=7)

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }

def create_model(input_vector_length=128):
    """
    Создаёт и компилирует нейросетевую модель для классификации пола.
    Использует полносвязные слои и Dropout для регуляризации.
    """
    model = Sequential()

    # Входной слой + Dropout
    model.add(Dense(256, input_shape=(input_vector_length,)))
    model.add(Dropout(0.3))

    # Несколько скрытых слоев с активацией ReLU и Dropout
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    # Выходной слой с активацией sigmoid для бинарной классификации
    model.add(Dense(1, activation="sigmoid"))

    # Компиляция модели с бинарной кроссэнтропией и метрикой точности
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()

    return model