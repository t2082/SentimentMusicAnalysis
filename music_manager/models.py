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




def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

# Root mean square
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)

# Mel-Frequency Cepstral coefficient
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

# Combine all feature functions
def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])

    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

# Apply data augmentation and extract its features
def get_features(path,duration=28, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset,mono=True)
    aud=extract_features(data)
    audio=np.array(aud)
    return audio

def preprocessing(path):
    # audio_data, sample_rate = librosa.load(path, sr=None)
    # mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # resized_mel = tf.image.resize(mel_spectrogram_db[..., np.newaxis], size=(128, 128))
    # resized_mel = np.repeat(resized_mel.numpy(), 3, axis=-1)
    # return resized_mel
    import numpy as np
    X=[]
    features=get_features(path)
    for i in features:
            X.append(i)
    X = np.array(X)
    X = X.reshape(1, X.shape[0], 1)
    return X

import joblib
def predict_audio(music, model='music_manager\emotional_music_classifier_model.h5', encoder_ = joblib.load('music_manager\encoder.pkl')):
    model = load_model(model)
    processed_data = preprocessing(music)
    prediction = model.predict(processed_data)
    pred_enc = encoder_.inverse_transform(prediction)
    class_ = pred_enc.flatten()[0]
    
    class_labels = ['dynamic', 'happy', 'sad', 'relaxed', 'anxious']
    for i, pred_proba in enumerate(prediction):
        pre_ = {genre: f"{proba * 100:.2f}" for genre, proba in zip(class_labels, pred_proba)}
    
    print(class_)
    
    if class_ == 1:
        predicted_genre = 'dynamic'
    elif class_ == 2:
        predicted_genre ='happy'
    elif class_ == 3:
        predicted_genre = 'sad'
    elif class_ == 4:
        predicted_genre ='relaxed'
    else:
        predicted_genre = 'anxious'
        
    print(pre_) 
    print(predicted_genre)
    
    return predicted_genre, pre_


def split_audio(audio_path, duration=29):
    y, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    audio_segments = []
    for i in range(0, int(total_duration), duration):
        start = i * sr
        end = (i + duration) * sr
        segment = y[start:end]
        if len(segment) < duration * sr:
            continue
        audio_segments.append(segment)
    return audio_segments

import numpy as np
import librosa

# def predict_audio(audio_data, model='music_manager\emotional_music_classifier_model.h5', encoder_=joblib.load('music_manager\encoder.pkl')):
#     model = load_model(model)
    
#     # Preprocess the audio data
#     processed_data = preprocessing(audio_data)
    
#     # Predict the emotional content
#     prediction = model.predict(processed_data)
#     pred_enc = encoder_.inverse_transform(prediction)
#     class_ = pred_enc.flatten()[0]
    
#     class_labels = ['dynamic', 'happy', 'sad', 'relaxed', 'anxious']
#     pre_ = {genre: 0 for genre in class_labels}  # Initialize pre_ with all classes having probability 0
    
#     for i, pred_proba in enumerate(prediction):
#         for j, proba in enumerate(pred_proba):
#             pre_[class_labels[j]] += proba * 100  # Add probability to each class
    
#     total_sum = sum(pre_.values())  # Calculate total probability sum
#     pre_percentage = {genre: (proba / total_sum) * 100 for genre, proba in pre_.items()}  # Calculate overall percentage
    
#     print(class_)
    
#     if class_ == 1:
#         predicted_genre = 'dynamic'
#     elif class_ == 2:
#         predicted_genre ='happy'
#     elif class_ == 3:
#         predicted_genre = 'sad'
#     elif class_ == 4:
#         predicted_genre ='relaxed'
#     else:
#         predicted_genre = 'anxious'
    
#     print(pre_percentage) 
#     print(predicted_genre)
    
#     return predicted_genre, pre_percentage


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