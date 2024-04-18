from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.urls import re_path
from django.views.static import serve
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_music, name='upload_music'),
    path('category/', views.category, name='music_category'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', views.register, name='register'),
    path('lyrics-classifier/', views.lyrics_page, name='lyric_classifier_page'),
    path('predict-lyrics/', views.predict_lyrics, name='lyric_classifier_predict'),
]

 