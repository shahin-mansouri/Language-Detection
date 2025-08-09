from django.urls import path
from .views import LanguageDetectAPIView, SupportedLanguagesAPIView

urlpatterns = [
    path('detect/', LanguageDetectAPIView.as_view(), name='api_detect'),
    path('languages/', SupportedLanguagesAPIView.as_view(), name='api_languages'),
]
