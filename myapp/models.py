from django.db import models

# Create your models here.

class Participant(models.Model):
    # participant form fields
    length = models.CharField(max_length=100)
    height = models.CharField(max_length=100)



    class Meta:
        db_table = "participants"