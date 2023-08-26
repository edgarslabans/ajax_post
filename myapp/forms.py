from .models import Participant
from django import forms
import datetime


class ParticipantForm(forms.ModelForm):
    ## set the label name of the date field.
    # = forms.DateField()
    rez = forms.CharField(widget=forms.HiddenInput())


    class Meta:
        model = Participant
        fields = ("__all__")