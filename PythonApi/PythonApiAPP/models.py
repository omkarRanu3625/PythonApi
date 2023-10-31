from django.db import models

class PythonApiModel(models.Model):
    DomainUUID = models.CharField(max_length=100, unique=True)
    wav_file = models.FileField(upload_to='uploads/')
    domain_name = models.CharField(max_length=100)

# Create models  for machnine learning emotion 
class Emotion(models.Model):
    emotion_type = models.CharField(max_length=50)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
