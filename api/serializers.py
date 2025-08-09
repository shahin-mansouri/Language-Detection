from rest_framework import serializers

class LanguageDetectionInputSerializer(serializers.Serializer):
    text = serializers.CharField()

class LanguageDetectionResultSerializer(serializers.Serializer):
    detected_language = serializers.CharField()
    confidence = serializers.FloatField()
    all_probabilities = serializers.DictField(child=serializers.FloatField())
    processing_time = serializers.FloatField()
    text_length = serializers.IntegerField()
    word_count = serializers.IntegerField()

class SupportedLanguageSerializer(serializers.Serializer):
    code = serializers.CharField()
    name = serializers.CharField()
    native_name = serializers.CharField()