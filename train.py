"""
В этом файле я реализую полный цикл обучения модели: от загрузки и разделения данных до обучения, валидации и сохранения модели.
Я также подключила EarlyStopping и TensorBoard для контроля переобучения и отслеживания метрик в процессе обучения.
"""

import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import load_data, split_data, create_model

# Загружаем признаки и метки пола из подготовленных данных
features, labels = load_data()

# Делим данные на обучающую, валидационную и тестовую выборки
datasets = split_data(features, labels, test_size=0.1, valid_size=0.1)

# Создаём модель нейросети (архитектура описана в utils.py)
network = create_model()

# Логируем обучение для визуализации в TensorBoard
tensorboard_cb = TensorBoard(log_dir="logs")

# Останавливаем обучение, если модель не улучшается на валидации в течение 5 эпох
early_stop_cb = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

# Параметры обучения
batch_sz = 64
num_epochs = 100

# Запускаем обучение с передачей данных и колбеков
network.fit(
    datasets["X_train"],
    datasets["y_train"],
    epochs=num_epochs,
    batch_size=batch_sz,
    validation_data=(datasets["X_valid"], datasets["y_valid"]),
    callbacks=[tensorboard_cb, early_stop_cb]
)

# Сохраняем обученную модель в папку results для дальнейшего использования
network.save("results/model.h5")

# Оцениваем качество модели на тестовой выборке
print(f"Оцениваю модель на {len(datasets['X_test'])} тестовых примерах...")
test_loss, test_acc = network.evaluate(datasets["X_test"], datasets["y_test"], verbose=0)

print(f"Финальная потеря (Loss): {test_loss:.4f}")
print(f"Точность на тесте (Accuracy): {test_acc * 100:.2f}%")
