from django import forms

class MusicForm(forms.Form):
    music_file = forms.FileField()
    
class EmotionForm(forms.Form):
    EMOTION_CHOICES = [
        ('dynamic', 'Dynamic'),
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('relaxed', 'Relax'),
        ('anxious', 'Anxious')
    ]
    emotion = forms.ChoiceField(choices=EMOTION_CHOICES)