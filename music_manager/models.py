from django.db import models    
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.image import resize
from tensorflow.keras.models import load_model
import re
from keras.models import model_from_json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Song(models.Model):
    title = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    file_path = models.FileField(upload_to='songs/')


def preprocessing(path):
    # audio_data, sample_rate = librosa.load(path, sr=None)
    # mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    # mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), size=(128, 128))
    audio_data, sample_rate = librosa.load(path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    resized_mel = tf.image.resize(mel_spectrogram_db[..., np.newaxis], size=(128, 128))
    resized_mel = np.repeat(resized_mel.numpy(), 3, axis=-1)
    return resized_mel
def predict_audio(music, model='music_manager\model.keras'):
    model = load_model(model)
    processed_data = preprocessing(music)
    processed_data = np.expand_dims(processed_data, axis=0)  # Ensure the input has the right shape
    prediction = model.predict(processed_data)[0]  # Get predictions for the batch
    genres = ['happy', 'sad', 'calm', 'nervous']
    predictions = {genre: f"{round(prob * 100, 2)}" for genre, prob in zip(genres, prediction)} 
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre, predictions

# Xử lí lời nhạc

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', 
 'Đk', 'Lyrics']

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x

def text_normalize(text):
    """Chuyển đổi văn bản sang chữ thường và loại bỏ dấu câu."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def remove_1_length_word(x):
    list = []
    for i in x:
         if len(i)> 1:
            list.append(i)
    return list
def preprocess_lyrics(lyrics):
    lyrics = text_normalize(lyrics)
    # lyrics = remove_accents(lyrics)
    lyrics = clean_text(lyrics)
    lyrics = clean_numbers(lyrics)
    lower_text = lambda x: [i.lower() for i in x.split(" ")]
    lyrics = lower_text(lyrics)
    filter_text = lambda x: " ".join(remove_1_length_word(x))
    lyrics = filter_text(lyrics)
    return lyrics

def predict_lyrics_(lyrics, model_path='music_manager\\biLSTM_w2v4.h5'):
    model = load_model(model_path)
    with open("music_manager\\tokenizer500.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    lyrics = preprocess_lyrics(lyrics)
    seq = tokenizer.texts_to_sequences([lyrics])
    padded = pad_sequences(seq, maxlen=500)
    prediction = model.predict(padded)
    percent_positive = prediction[0][1] * 100
    percent_negative = prediction[0][0] * 100
    predicted_class = np.argmax(prediction, axis=1)
    print('debug class: ', predicted_class)
    print('debug percent_positive: ', percent_positive)
    print('debug percent_negative: ', percent_negative)
    return predicted_class, percent_positive, percent_negative