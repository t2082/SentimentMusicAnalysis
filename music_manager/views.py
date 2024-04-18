from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
import os
from django.http import HttpResponse
from .forms import MusicForm
from .models import predict_audio
from .forms import EmotionForm
from .models import predict_lyrics_
import random
import shutil
import re

def home(request):
    return render(request, 'home.html')

def lyrics_page (request):
    return render(request, 'lyrics_classifier.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def upload_music(request):
    if request.method == 'POST':
        form = MusicForm(request.POST, request.FILES)
        if form.is_valid():
            music_file = request.FILES['music_file']
            music_file_path = handle_uploaded_file(music_file)
            genre, predictions = predict_audio(music_file_path)
            emo_path = f'local_database/{genre}'
            if not os.path.exists(emo_path):
                os.makedirs(emo_path)
            if not os.path.exists(f'local_database/{music_file.name}'):
                shutil.move(f'local_database/{music_file.name}', emo_path)
            context = {'song_name': music_file.name, 'predictions': predictions}
            return render(request, 'home.html', context)
    else:
        form = MusicForm()
    return render(request, 'index.html', {'form': form})


def predict_lyrics(request):
    if request.method == 'POST':
        lyrics = request.POST.get('lyricsInput')
        sentences = re.split(r'[.?!;\n]\s*', lyrics.strip())
        count_positive = 0
        count_negative = 0

        for sentence in sentences:
            if sentence:
                result = predict_lyrics_(sentence)  # Gọi hàm dự đoán của bạn
                if result == 1:
                    count_positive += 1
                else:
                    count_negative += 1

        total_sentences = count_positive + count_negative  # Tổng số câu được phân tích
        if total_sentences > 0:  # Kiểm tra để tránh chia cho zero
            percent_positive = (count_positive / total_sentences) * 100
            percent_negative = (count_negative / total_sentences) * 100
        else:
            percent_positive = 0
            percent_negative = 0

        # Xuất kết quả tỷ lệ phần trăm và phân loại
        if count_positive > count_negative:
            result = "Tích cực"
        elif count_negative > count_positive:
            result = "Tiêu cực"
        else:
            result = "Trung tính"

        # Trả về kết quả cùng tỷ lệ phần trăm trong template
        return render(request, 'lyrics_classifier.html', {
            'result': result,
            'percent_positive': f'{percent_positive:.2f}%',
            'percent_negative': f'{percent_negative:.2f}%'
        })


def handle_uploaded_file(f):
    save_path = 'local_database/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, f.name)
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path


def category(request):
    files = []
    emotion = ''
    if request.method == 'POST':
        form = EmotionForm(request.POST)
        if form.is_valid():
            emotion = form.cleaned_data['emotion']
            directory = f'local_database/{emotion}/'

            all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            if len(all_files) >= 12: 
                files = random.sample(all_files, 12) 
            else:
                files = all_files 
    else:
        form = EmotionForm()
    return render(request, 'music_category.html', {'form': form, 'files': files, 'emotion': emotion})