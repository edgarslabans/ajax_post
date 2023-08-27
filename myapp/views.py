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
    if request.method == "POST":  # request.is_ajax  - does not work
        # get the form data
        form = ParticipantForm(request.POST)

        # adjusting request with new data
        post = request.POST.copy()  # to make it mutable
        post['height'] = add_something(post['length'])
        form = ParticipantForm(post)


        #print("type", type(form), form['length'].value(), "is valid", form.is_valid)

        # save the data and after fetch the object in instance
        if form.is_valid():
            instance = form.save()
            # print(type(instance), instance)
            # serialize in new participant object in json
            ser_instance = serializers.serialize('json', [instance, ])
            # send to client side.

            #print("ttttt", {"instance": ser_instance})
            return JsonResponse({"instance": ser_instance}, status=200)
            # return JsonResponse({"instance": {"first_name": "bar","last_name": "2bar"}}, status=200)

        else:
            # some form errors occured.
            return JsonResponse({"error": form.errors}, status=400)

    # some error occured+

    return JsonResponse({"error": ""}, status=400)


def conv_to_dict(inp):
    shorter_inp = inp[inp.find('{') + 1: inp.rfind('}')]
    shorter_inp2 = shorter_inp[shorter_inp.find('{') + 1: shorter_inp.rfind('}')]
    dict2 = json.loads('{' + shorter_inp2 + '}')
    # dict2 = shorter_inp2
    return dict2
