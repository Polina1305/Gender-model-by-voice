"""
Скрипт для предобработки и извлечения аудиофичей из набора Common Voice (Mozilla)

Этот скрипт я использовала для подготовки данных из датасета Mozilla Common Voice
(https://www.kaggle.com/datasets/mozillaorg/common-voice) для задачи распознавания пола по голосу.

Функциональность скрипта:
- Загружает .csv-файлы с метаинформацией (имя файла, пол)
- Фильтрует только записи с метками "male" и "female"
- Извлекает аудиофичи из .wav-файлов с помощью `librosa` (по умолчанию — mel-спектрограммы)
- Сохраняет признаки в формате `.npy` для последующей загрузки в модель

Результатом работы является папка `data/`, содержащая отфильтрованные .csv и `.npy`-файлы с признаками аудио.
"""



# https://www.kaggle.com/datasets/mozillaorg/common-voice
# https://thepythoncode.com/article/gender-recognition-by-voice-using-tensorflow-in-python
# https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html


import glob
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm


def extract_audio_features(file_path, **kwargs):
    """
    Извлекает аудиофичи из файла file_path.
    """
    use_mfcc = kwargs.get("mfcc")
    use_chroma = kwargs.get("chroma")
    use_mel = kwargs.get("mel")
    use_contrast = kwargs.get("contrast")
    use_tonnetz = kwargs.get("tonnetz")

    y, sr = librosa.core.load(file_path)
    result = np.array([])

    if use_chroma or use_contrast:
        stft = np.abs(librosa.stft(y))

    if use_mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if use_chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_feat))
    if use_mel:
        mel_spec = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
        result = np.hstack((result, mel_spec))
    if use_contrast:
        contrast_feat = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, contrast_feat))
    if use_tonnetz:
        tonnetz_feat = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        result = np.hstack((result, tonnetz_feat))

    return result


output_folder = "data"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

csv_list = glob.glob("*.csv")

for idx, csv_name in enumerate(csv_list):
    print("[+] Обработка:", csv_name)
    df = pd.read_csv(csv_name)
    filtered_df = df[["filename", "gender"]]
    print("Исходных строк:", len(filtered_df))

    # Убираем лишние записи
    filtered_df = filtered_df[filtered_df["gender"].isin(["male", "female"])]
    print("После фильтрации:", len(filtered_df))

    clean_csv = os.path.join(output_folder, csv_name)
    filtered_df.to_csv(clean_csv, index=False)

    folder_prefix, _ = csv_name.split(".")
    audio_paths = glob.glob(f"{folder_prefix}/{folder_prefix}/*")
    valid_audio_files = set(filtered_df["filename"])

    for idx, audio_path in tqdm(list(enumerate(audio_paths)), f"Обработка {folder_prefix}"):
        parent_folder, file_name = os.path.split(audio_path)
        full_name = f"{os.path.basename(parent_folder)}/{file_name}"

        if full_name in valid_audio_files:
            from_path = f"{folder_prefix}/{full_name}"
            to_path = f"{output_folder}/{full_name}"
            target_folder = os.path.dirname(to_path)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            features = extract_audio_features(from_path, mel=True)
            save_name = to_path.rsplit(".", 1)[0]
            np.save(save_name, features)
