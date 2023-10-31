# serializers.py

from rest_framework import serializers
from .models import PythonApiModel

class MyModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = PythonApiModel
        fields = ('DomainUUID', 'wav_file', 'domain_name')

# Code for dirct json analyzer
class AnalyzeAudioRequestSerializer(serializers.Serializer):
    domain_name = serializers.CharField()
    file_path = serializers.CharField()
    domain_uuid = serializers.CharField()
