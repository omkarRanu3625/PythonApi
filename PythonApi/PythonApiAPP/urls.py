from django.urls import path
from .views import create_record,fetch_and_transcribe_audio, analyze_audio, analyze_live_audio, analyze_audio1, analyze_audio3,EmotionDetectionAPI

urlpatterns = [
    path('create/', create_record.as_view(), name='create-record'),
    path('fetch-and-transcribe/', fetch_and_transcribe_audio, name='fetch-and-transcribe'),
    path('analyze-audio/', analyze_audio, name='analyze-audio'),
    path('analyze-live-audio/', analyze_live_audio, name='analyze-live-audio'),
    path('analyze-audio1/', analyze_audio1, name='analyze-audio1'),
    # path('recognize-speech/', recognize_speech, name='recognize-speech'),
    path('analyze-audio3/', analyze_audio3, name='analyze-audio3'),
    path('detect-emotion/', EmotionDetectionAPI.as_view(), name='detect-emotion'),
]