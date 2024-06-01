from django.contrib import admin
from django.urls import path
from .views import Summarization
from .views import translate_english_to_arabic
from .views import DocumentUploadView

urlpatterns = [
    path('summarize/', Summarization),
    path('translate/', translate_english_to_arabic),
    path('upload/', DocumentUploadView.as_view(), name='file-upload')
    
]
