from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from detection.language_detector import LanguageDetector
from detection.models import Language
from .serializers import LanguageDetectionInputSerializer, LanguageDetectionResultSerializer, SupportedLanguageSerializer

# Create your views here.

class LanguageDetectAPIView(APIView):
    def post(self, request):
        serializer = LanguageDetectionInputSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            detector = LanguageDetector()
            import time
            start_time = time.time()
            result = detector.predict_with_confidence(text)
            processing_time = time.time() - start_time
            # Try to get language name from DB
            lang_obj = Language.objects.filter(code=result['predicted_language']).first()
            detected_language = lang_obj.name if lang_obj else result['predicted_language']
            output = {
                'detected_language': detected_language,
                'confidence': round(result['confidence'] * 100, 1),
                'all_probabilities': {k: round(v * 100, 1) for k, v in result['all_probabilities'].items()},
                'processing_time': processing_time,
                'text_length': result['text_length'],
                'word_count': len(text.split()),
            }
            return Response(output)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SupportedLanguagesAPIView(APIView):
    def get(self, request):
        languages = Language.objects.filter(is_active=True)
        serializer = SupportedLanguageSerializer(languages, many=True)
        return Response(serializer.data)
