import uuid
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import PythonApiModel,Emotion
from .serializers import MyModelSerializer, AnalyzeAudioRequestSerializer
import os
import json
import speech_recognition as sr
from rest_framework.decorators import api_view
import requests
from pydub import AudioSegment
import io
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import websockets
import asyncio
from asgiref.sync import async_to_sync
from textblob import TextBlob
import pyaudio
from django.http import JsonResponse
import requests
# from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
import os
import sounddevice as sd
import numpy as np

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import numpy as np
import tensorflow as tf
import librosa


#/* Upload Record on Database */
class create_record(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        mutable_data = request.data.copy()  # Create a mutable copy of request.data
        mutable_data['DomainUUID'] = str(uuid.uuid4())  # Generate auto-generated DomainUUID as a string
        serializer = MyModelSerializer(data=mutable_data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
# audio ffetch and transcripts in text 

@api_view(['POST'])
def fetch_and_transcribe_audio(request):
    domain_uuid = request.data.get('DomainUUID')

    try:
        audio_file = PythonApiModel.objects.get(DomainUUID=domain_uuid).wav_file.path
        transcript = transcribe_audio(audio_file)
        response_data = {
            'DomainUUID': domain_uuid,
            'Transcript': transcript
        }
        return Response(response_data, status=200)
    except PythonApiModel.DoesNotExist:
        return Response({'error': 'Audio file not found for the given DomainUUID'}, status=404)

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
            return transcript
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        
# direct sentiment analysis
@api_view(['POST'])
def analyze_audio(request):
    if request.method == 'POST':
        try:
            serializer = AnalyzeAudioRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            data = serializer.validated_data

            # Construct the audio URL
            audio_url = f"https://{data['domain_name']}/{data['file_path']}/{data['domain_uuid']}.wav"

            # Fetch audio content from the URL
            response = requests.get(audio_url)
            audio_data = response.content

            # Convert audio data to AudioSegment
            song = AudioSegment.from_wav(io.BytesIO(audio_data))

            chunk_duration = 5000
            overlap_duration = 200
            audio_duration = len(song)
            recognized_text_list = []

            start = 0
            end = chunk_duration

            while start < audio_duration:
                if end > audio_duration:
                    end = audio_duration

                chunk = song[start:end]

                try:
                    r = sr.Recognizer()
                    audio_data = sr.AudioData(chunk.raw_data, chunk.frame_rate, chunk.sample_width)
                    text = r.recognize_google(audio_data)

                    sentiment = SentimentIntensityAnalyzer()
                    sentiment_scores = sentiment.polarity_scores(text)
                    print(f"Sentiment: {sentiment_scores['compound']}")
                    #this for only text
                    recognized_text_list.append(text)

                except Exception as e:
                    print(f"Sentiment: 0.0: {e}")

                start += chunk_duration - overlap_duration
                end = start + chunk_duration

            recognized_text = ' '.join(recognized_text_list)
            print(recognized_text)
            return Response({'result': recognized_text})

            # # This for text with sentiment
            #         recognized_text_list.append({'text': text, 'sentiment': sentiment_scores['compound']})

            #     except Exception as e:
            #         print(f"Error: {e}")

            #     start += chunk_duration - overlap_duration
            #     end = start + chunk_duration

            # return Response({'result': recognized_text_list})
        except Exception as e:
            return Response({'error': str(e)}, status=500)
    else:
        return Response({'error': 'Invalid request method'}, status=400)

#live other api

@api_view(['POST'])
def analyze_audio3(request):
    if request.method == 'POST':
        try:
            serializer = AnalyzeAudioRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            data = serializer.validated_data

            # Sample rate and chunk size for streaming audio
            sample_rate = 44100  # You might need to adjust this based on the audio stream's sample rate
            chunk_size = 1024

            # Open a streaming connection to the live audio stream
            audio_stream_url = f"https://{data['domain_name']}/{data['file_path']}/{data['domain_uuid']}.wav"
            audio_stream = requests.get(audio_stream_url, stream=True)

            recognized_text_list = []
            # sentiment_analyzer = SentimentIntensityAnalyzer()

            with sd.InputStream(samplerate=sample_rate, channels=2, blocksize=chunk_size, dtype='int16') as stream:
                for chunk in audio_stream.iter_content(chunk_size=chunk_size):
                    if chunk:
                        # Convert chunk to numpy array
                        audio_data = np.frombuffer(chunk, dtype=np.int16)
                        
                        # Process audio data (you can perform additional audio processing here if needed)
                        # For example, you can convert audio_data to AudioSegment and perform text recognition
                        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
                        chunk_duration = 5000
                        overlap_duration = 200
                        audio_duration = len(audio_segment)
                        recognized_text_list = []

                        start = 0
                        end = chunk_duration

                        while start < audio_duration:
                            if end > audio_duration:
                                end = audio_duration

                            chunk = audio_segment[start:end]

                            try:
                                r = sr.Recognizer()
                                audio_data = sr.AudioData(chunk.raw_data, chunk.frame_rate, chunk.sample_width)
                                text = r.recognize_google(audio_data)

                                sentiment = SentimentIntensityAnalyzer()
                                sentiment_scores = sentiment.polarity_scores(text)
                                print(f"Sentiment: {sentiment_scores['compound']}")
                                #this for only text
                                recognized_text_list.append(text)
  
                            except Exception as e:
                                print(f"Sentiment: 0.0: {e}")

                            start += chunk_duration - overlap_duration
                            end = start + chunk_duration

                        recognized_text = ' '.join(recognized_text_list)
                        print(recognized_text)
                        return Response({'result': recognized_text})

            #             # Perform sentiment analysis on recognized text
            #             sentiment_scores = sentiment_analyzer.polarity_scores(recognized_text)
            #             print(f"Sentiment: {sentiment_scores['compound']}")
                        
            #             # Add recognized text and sentiment to the list
            #             recognized_text_list.append(recognized_text)

            # # Join recognized text chunks into a single string
            # recognized_text = ' '.join(recognized_text_list)
            # print(recognized_text)

            # return Response({'result': recognized_text})

        except Exception as e:
            return Response({'error': str(e)}, status=500)
    else:
        return Response({'error': 'Invalid request method'}, status=400)


# Create a api for live transcript
@api_view(['POST'])
# @async_to_sync
async def analyze_live_audio(request):
    if request.method == 'POST':
        try:
            serializer = AnalyzeAudioRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            data = serializer.validated_data

            # Create a unique session ID for this live call
            session_id = str(uuid.uuid4())

            async def audio_transcription(websocket, path):
                recognizer = sr.Recognizer()

                # Construct the audio URL
                audio_url = f"https://{data['domain_name']}/{data['file_path']}/{data['domain_uuid']}.wav"

                # Use websockets to stream audio data
                async with websockets.connect(audio_url) as websocket:
                    while True:
                        audio_data = await websocket.recv()
                        try:
                            with io.BytesIO(audio_data) as audio_stream:
                                audio_stream.seek(0)
                                with sr.AudioFile(audio_stream) as source:
                                    audio = recognizer.record(source)  # Read the entire audio file
                                    try:
                                        recognized_text = recognizer.recognize_google(audio)
                                        sentiment = SentimentIntensityAnalyzer()
                                        sentiment_scores = sentiment.polarity_scores(recognized_text)
                                        print(f"Sentiment: {sentiment_scores['compound']}")
                                        response_data = {'text': recognized_text, 'sentiment': sentiment_scores['compound']}
                                    except sr.UnknownValueError:
                                        response_data = {'text': '', 'sentiment': 0.0}
                                    except sr.RequestError as e:
                                        response_data = {'text': '', 'sentiment': 0.0}
                                        print(f"Error: {e}")
                                    await websocket.send(json.dumps(response_data))

                        except Exception as e:
                            print(f"Error: {e}")
                            await websocket.send(json.dumps({'text': '', 'sentiment': 0.0}))

            # Start the WebSocket server asynchronously
            start_server = websockets.serve(audio_transcription, '0.0.0.0', 8765)
            print("WebSocket server started...")
            await start_server

            return Response({'message': 'Live call transcription started.'})

        except Exception as e:
            return Response({'error': str(e)}, status=500)
    else:
        return Response({'error': 'Invalid request method'}, status=400)

# Create your views here.
@api_view(['POST'])
def analyze_audio1(request):
    if request.method == 'POST':
        try:
            serializer = AnalyzeAudioRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            data = serializer.validated_data

            # Construct the audio URL (assuming a streaming API endpoint)
            audio_url = f"https://{data['domain_name']}/{data['file_path']}/{data['domain_uuid']}.wav"

            # Initialize recognizer and microphone instances
            r = sr.Recognizer()
            mic = sr.Microphone()

            # Listen to the live audio stream in chunks of 10 seconds
            chunk_duration = 10  # in seconds
            overlap_duration = 2  # in seconds

            recognized_text_list = []

            with mic as source:
                while True:
                    audio_data = r.listen(source, timeout=chunk_duration + overlap_duration, phrase_time_limit=chunk_duration)

                    try:
                        # Recognize the audio chunk to text
                        text = r.recognize_google(audio_data)

                        # Perform sentiment analysis on the recognized text
                        sentiment = TextBlob(text).sentiment.polarity
                        print(f"Sentiment: {sentiment}")

                        recognized_text_list.append({'text': text, 'sentiment': sentiment})

                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio")

                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")

                    except Exception as e:
                        print(f"Error: {e}")

                    if len(audio_data.frame_data) == 0:
                        break

            return Response({'result': recognized_text_list})

        except Exception as e:
            return Response({'error': str(e)}, status=500)

    else:
        return Response({'error': 'Invalid request method'}, status=400)
    
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/omkar/Python/OfficeWork/private_key.json'
# Create another using google 
# def recognize_speech(request):
#     if request.method == 'POST':
#         data = request.POST
#         audio_url = data.get('audio_url')
#         client = speech.SpeechClient()

#         config = types.RecognitionConfig(
#             encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=16000,
#             language_code='en-US',
#         )
#         audio = types.RecognitionAudio(uri=audio_url)
#         response = client.recognize(config=config, audio=audio)
#         transcript = ""
#         for result in response.results:
#             transcript += result.alternatives[0].transcript + " "
#         return JsonResponse({'transcript': transcript})
#     else:
#         return JsonResponse({'error': 'Invalid request method'})


# Machine learning model emotions
class EmotionDetectionAPI(View):
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def post(self, request):
        try:
            audio_file_path = json.loads(request.body)['audio']  # Assuming 'audio' is the key in JSON containing audio data
            audio_data, _ = librosa.load(audio_file_path,res_type='kaiser_fast',duration=3, offset=0.5)
            audio_array = np.array(audio_data)
            # Extract features from audio data using the extract_features function
            features = self.extract_features(audio_array)
            features = features.reshape(1, features.shape[0], features.shape[1])  # Reshape for model input
            # Load the pre-trained model
            model = tf.keras.models.load_model('/home/omkar/go/src/PythonAPI/PythonApi/Speech-Emotion-Analyzer/saved_models/Emotion_Voice_Detection_Model.h5')
            
            # Predict emotion
            emotion_label = self.predict_emotion(model, audio_array)
            confidence = 0.95  # Set confidence level as needed
            
            # Save emotion prediction to database
            Emotion.objects.create(emotion_type=emotion_label, confidence=confidence)
            
            return JsonResponse({'success': True, 'message': 'Emotion detected and saved successfully.'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    @staticmethod
    def predict_emotion(model, audio_array):
        # Perform necessary preprocessing on audio_array (e.g., convert to spectrogram)
        # Make sure the audio_array has the same format as the input data used for training the model
        
        # Perform prediction
        predicted_label = model.predict(np.expand_dims(audio_array, axis=0))
        
        # Decode the predicted label (use the mapping provided in the README.md)
        decoded_label = EmotionDetectionAPI.decode_label(np.argmax(predicted_label))
        return decoded_label
    
    @staticmethod
    def extract_features(audio_data, sample_rate=44100, n_mels=20, n_fft=2048, hop_length=512, num_segments=10):
        # Divide the audio into segments
        segment_size = len(audio_data) // num_segments
        spectrogram_features = []
    
        for segment in range(num_segments):
            start = segment * segment_size
            end = start + segment_size
        
            # Compute the spectrogram for each segment
            spectrogram = librosa.feature.melspectrogram(y=audio_data[start:end], sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
            # Convert to log scale (dB)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
            # Ensure the spectrogram has at least 10 time steps
            if spectrogram_db.shape[1] < 10:
                # If the number of time steps is less than 10, pad the spectrogram
                pad_width = 10 - spectrogram_db.shape[1]
                spectrogram_db = np.pad(spectrogram_db, pad_width=((0, 0), (0, pad_width)))
        
            # Take the first 10 time steps if there are more
            spectrogram_db = spectrogram_db[:, :10]
        
            # Reshape the spectrogram to match the expected input shape (1, 10, 13)
            spectrogram_db = spectrogram_db.reshape((1, 20, 1))
        
            spectrogram_features.append(spectrogram_db)
    
        return np.array(spectrogram_features)
    
    @staticmethod
    def decode_label(label):
        # Decode label based on the provided mapping
        # Add your mapping logic here
        mapping = {
            0: 'female_angry',
            1: 'female_calm',
            2: 'female_fearful',
            3: 'female_happy',
            4: 'female_sad',
            5: 'male_angry',
            6: 'male_calm',
            7: 'male_fearful',
            8: 'male_happy',
            9: 'male_sad'
        }
        return mapping[label]
    
