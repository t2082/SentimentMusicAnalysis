from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Function to preprocess text
import re
from underthesea import word_tokenize
import pickle


num_classes = 2

# Number of dimensions for word embedding
embed_num_dims = 300

# Max input length (max number of words) 
max_seq_len= 300

class_names = ['0','1']

from keras.models import model_from_json


# Tải cấu trúc mô hình từ JSON
with open("D:\\EmotionalMusicClassifier\\music_project\\music_manager\\biLSTM_structure.json", 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Tải trọng số mô hình
model.load_weights("D:\\EmotionalMusicClassifier\\music_project\\music_manager\\biLSTM.h5")
# Load the model
# model = load_model("D:\\EmotionalMusicClassifier\\music_project\\music_manager\\biLSTM.h5")

# Text to predict

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

text = "Nhìn ngán vl"
text = text_normalize(text)
text = remove_accents(text)
text = clean_text(text)
text = clean_numbers(text)
lower_text = lambda x: [i.lower() for i in x.split(" ")]
text = lower_text(text)
filter_text = lambda x: " ".join(remove_1_length_word(x))
text = filter_text(text)
print(text)
# Tải tokenizer
with open("D:\\EmotionalMusicClassifier\\music_project\\music_manager\\tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)
seq = tokenizer.texts_to_sequences([text])
padded = pad_sequences(seq, maxlen=300)
# Predict
prediction = model.predict(padded)
predicted_class = np.argmax(prediction, axis=1)
print("Predicted class:", predicted_class)
