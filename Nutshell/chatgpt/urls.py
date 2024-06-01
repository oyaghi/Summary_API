from django.contrib import admin
from django.urls import path
from .views import QuestionAnswerView

urlpatterns = [
    path('get-answer/', QuestionAnswerView.as_view()),
    
]
