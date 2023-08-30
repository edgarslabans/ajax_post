from django.forms import TextInput

from .models import Calculation
from django import forms
import datetime


class ParticipantForm(forms.ModelForm):
    ## set the label name of the date field.
    # = forms.DateField()

    EI = forms.CharField(initial="6369")
    GA = forms.CharField(max_length=100,initial="500")

    L = forms.CharField(max_length=100, initial="0/5/0/0")

    LP1 = forms.CharField(max_length=100, initial="2@3")
    LP2 = forms.CharField(max_length=100, initial="6@5")

    q_load = forms.IntegerField( initial=100)


    class Meta:
        model = Calculation
        fields = ("__all__")

