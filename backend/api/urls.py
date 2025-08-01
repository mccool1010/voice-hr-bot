from django.urls import path
from .views import hr_bot

urlpatterns = [
    path('chat/', hr_bot),
]
