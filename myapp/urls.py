from django.contrib import admin
from django.urls import path
from myapp.views import (
    displayData,
    postParticipant,
)

urlpatterns = [
    path('', displayData),
    path('post/ajax/participant', postParticipant, name = "post_participant"),
]