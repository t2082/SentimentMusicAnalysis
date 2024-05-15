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
            emo_path = f'local_database/{genre}/'
            if not os.path.exists(emo_path):
                os.makedirs(emo_path)
            if os.path.exists(f'local_database/{music_file.name}'):
                if not os.path.exists(f'{emo_path}/{music_file.name}'):
                    shutil.move(f'local_database/{music_file.name}', emo_path)
                else:
                    os.remove(f'local_database/{music_file.name}')
            context = {'song_name': music_file.name, 'predictions': predictions}
            return render(request, 'home.html', context)
    else:
        form = MusicForm()
    return render(request, 'index.html', {'form': form})


def predict_lyrics(request):
    if request.method == 'POST':
        lyrics = request.POST.get('lyricsInput')
        if str(lyrics) == '':
            return render(request, 'lyrics_classifier.html', {
            'lyrics': str(lyrics),
            'result': 'Please input a lyrics !',
            'percent_positive': f'{50.00}',
            'percent_negative': f'{50.00}'
        })

        result, percent_positive, percent_negative = predict_lyrics_(lyrics)  # Gọi hàm dự đoán của bạn
        if result == 1:
            result = "Positive"
        else:
            result = "Negative"
        # Trả về kết quả cùng tỷ lệ phần trăm trong template
        return render(request, 'lyrics_classifier.html', {
            'lyrics': str(lyrics),
            'result': result,
            'percent_positive': f'{percent_positive:.2f}',
            'percent_negative': f'{percent_negative:.2f}'
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