from django.shortcuts import render
from django.http import JsonResponse
from django.core import serializers
from .forms import ParticipantForm
from .models import Participant
from .core_calculations import add_something

import json

def displayData(request):
    form = ParticipantForm()
    participants = Participant.objects.all()
    return render(request, "index.html", {"form": form, "participants": participants})

def postParticipant(request):
    # request should be ajax and method should be POST.
    if  request.method == "POST":                              #request.is_ajax  - does not work
        # get the form data
        form = ParticipantForm(request.POST)

        # save the data and after fetch the object in instance
        if form.is_valid():
            instance = form.save()
            #print(type(instance), instance)
            # serialize in new participant object in json
            ser_instance = serializers.serialize('json', [ instance, ])
            # send to client side.
            f_name = conv_to_dict(ser_instance)["first_name"]

            # find the value of the first_name
            rezz = ser_instance[ser_instance.find('first_name')+len('first_name')+4: ser_instance.find('last_name')-4]


            x = ser_instance.replace(rezz, add_something("Custom input"))
            #print("rezz", rezz, len(rezz),x)
            return JsonResponse({"instance": x}, status=200)
        else:
            # some form errors occured.
            return JsonResponse({"error": form.errors}, status=400)

    # some error occured
    return JsonResponse({"error": ""}, status=400)


def conv_to_dict(inp):

    shorter_inp = inp[inp.find('{')+1: inp.rfind('}')]
    shorter_inp2 = shorter_inp[shorter_inp.find('{') + 1: shorter_inp.rfind('}')]
    dict2 = json.loads('{'+shorter_inp2+'}')
    #dict2 = shorter_inp2
    return dict2


