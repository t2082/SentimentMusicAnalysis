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
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x

s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

def text_normalize(text):
    """Chuyển đổi văn bản sang chữ thường và loại bỏ dấu câu."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_accents(input_str):
  s = ''
  for c in input_str:
    if c in s1:
       s += s0[s1.index(c)]
    else:
      s += c
  return s


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

def predict_lyrics_(lyrics, model_path='music_manager\\biLSTM.h5'):
    with open("music_manager\\biLSTM_structure.json", 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(model_path)
    with open("music_manager\\tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    lyrics = preprocess_lyrics(lyrics)
    seq = tokenizer.texts_to_sequences([lyrics])
    padded = pad_sequences(seq, maxlen=300)
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction, axis=1)
    print('debug class: ', predicted_class)
    return predicted_class