"""
Этот скрипт я разработала для распознавания пола человека на основе его голосовой записи.  
Он может работать как с предварительно записанными .wav-файлами,
так и с записью голоса в реальном времени через микрофон.

Функциональность скрипта:
- Запись аудио с микрофона и предварительная очистка сигнала
- Выделение аудиофичей (MFCC, Mel, Chroma и др.)
- Предсказание пола на основе обученной модели
- Поддержка командной строки (можно указать путь к файлу через `--file`)

Это удобный и расширяемый инструмент, который может использоваться и в реальном времени, и как часть более крупной системы.
"""


import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack

# Порог громкости, ниже которого считается тишина
SILENCE_THRESHOLD = 500
# Размер блока для чтения аудио
BLOCK_SIZE = 1024
# Формат аудио - 16-битный PCM
AUDIO_FORMAT = pyaudio.paInt16
# Частота дискретизации (Гц)
SAMPLE_RATE = 16000

# Кол-во подряд идущих блоков тишины, чтобы остановить запись
MAX_SILENCE_BLOCKS = 30

def check_silence(sound_chunk):
    """Проверяет, является ли аудиоблок тишиной"""
    return max(sound_chunk) < SILENCE_THRESHOLD

def normalize_audio(sound_data):
    """Нормализует громкость записи, увеличивая сигнал до заданного максимума"""
    MAX_VOLUME = 16384
    multiplier = float(MAX_VOLUME) / max(abs(sample) for sample in sound_data)

    normalized = array('h')
    for sample in sound_data:
        normalized.append(int(sample * multiplier))
    return normalized

def trim_silence(sound_data):
    """Обрезает тишину с начала и конца аудиозаписи"""
    def trim_start_end(data):
        started = False
        trimmed = array('h')

        for sample in data:
            if not started and abs(sample) > SILENCE_THRESHOLD:
                started = True
                trimmed.append(sample)
            elif started:
                trimmed.append(sample)
        return trimmed

    sound_data = trim_start_end(sound_data)
    sound_data.reverse()
    sound_data = trim_start_end(sound_data)
    sound_data.reverse()
    return sound_data

def add_padding_silence(sound_data, seconds):
    """Добавляет тишину в начале и конце аудиозаписи"""
    silence_chunk = array('h', [0] * int(seconds * SAMPLE_RATE))
    silence_chunk.extend(sound_data)
    silence_chunk.extend([0] * int(seconds * SAMPLE_RATE))
    return silence_chunk

def record_audio():
    """
    Записывает аудио с микрофона.
    Автоматически начинает и заканчивает запись в зависимости от тишины.
    Возвращает нормализованные и обрезанные данные с добавленной тишиной.
    """
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE,
                                 input=True, output=True, frames_per_buffer=BLOCK_SIZE)

    silence_blocks = 0
    recording_started = False
    recorded_data = array('h')

    while True:
        data_chunk = array('h', stream.read(BLOCK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        recorded_data.extend(data_chunk)

        if check_silence(data_chunk):
            if recording_started:
                silence_blocks += 1
        else:
            if not recording_started:
                recording_started = True
                silence_blocks = 0

        if recording_started and silence_blocks > MAX_SILENCE_BLOCKS:
            break

    sample_width = audio_interface.get_sample_size(AUDIO_FORMAT)
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

    normalized = normalize_audio(recorded_data)
    trimmed = trim_silence(normalized)
    padded = add_padding_silence(trimmed, 0.5)
    return sample_width, padded

def save_recording_to_wav(filepath):
    """Записывает звук, записанный с микрофона, в WAV-файл"""
    sample_width, audio_data = record_audio()
    packed_data = pack('<' + ('h' * len(audio_data)), *audio_data)

    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(packed_data)

def extract_features_from_file(filepath, **kwargs):
    """
    Извлекает аудиофичи из файла с помощью librosa.
    Поддерживаются: mfcc, chroma, mel, contrast, tonnetz.
    """
    mfcc = kwargs.get('mfcc', False)
    chroma = kwargs.get('chroma', False)
    mel = kwargs.get('mel', False)
    contrast = kwargs.get('contrast', False)
    tonnetz = kwargs.get('tonnetz', False)

    signal, sr = librosa.core.load(filepath)

    if chroma or contrast:
        stft = np.abs(librosa.stft(signal))

    features_vector = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
        features_vector = np.hstack((features_vector, mfccs))
    if chroma:
        chroma_feats = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        features_vector = np.hstack((features_vector, chroma_feats))
    if mel:
        mel_feats = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T, axis=0)
        features_vector = np.hstack((features_vector, mel_feats))
    if contrast:
        contrast_feats = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        features_vector = np.hstack((features_vector, contrast_feats))
    if tonnetz:
        tonnetz_feats = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr).T, axis=0)
        features_vector = np.hstack((features_vector, tonnetz_feats))

    return features_vector


if __name__ == "__main__":
    import argparse
    from utils import create_model

    parser = argparse.ArgumentParser(description="Скрипт для распознавания пола по голосу. "
                                                 "Используйте файл или запись с микрофона.")
    parser.add_argument("-f", "--file", help="Путь к аудиофайлу в формате WAV")
    args = parser.parse_args()

    audio_file = args.file

    model = create_model()
    model.load_weights("results/model.h5")

    if not audio_file or not os.path.isfile(audio_file):
        print("Говорите, запись началась...")
        audio_file = "recorded_test.wav"
        save_recording_to_wav(audio_file)

    features = extract_features_from_file(audio_file, mel=True).reshape(1, -1)
    male_probability = model.predict(features)[0][0]
    female_probability = 1 - male_probability

    predicted_gender = "male" if male_probability > female_probability else "female"

    print("Определён пол:", predicted_gender)
    print(f"Вероятность: Мужской - {male_probability*100:.2f}%, Женский - {female_probability*100:.2f}%")