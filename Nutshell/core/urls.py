from django.contrib import admin
from django.urls import path
from .views import Summarization

urlpatterns = [
    path('summarize/', Summarization)
]
