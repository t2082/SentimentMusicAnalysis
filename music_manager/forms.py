from django import forms

class MusicForm(forms.Form):
    music_file = forms.FileField()
    
class EmotionForm(forms.Form):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('calm', 'Calm'),
        ('nervous', 'Nervous')
    ]
    emotion = forms.ChoiceField(choices=EMOTION_CHOICES)