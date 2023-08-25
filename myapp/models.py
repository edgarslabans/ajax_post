from django.db import models

# Create your models here.

class Participant(models.Model):
    # participant form fields
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    class Meta:
        db_table = "participants"