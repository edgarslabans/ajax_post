from django.db import models

# Create your models here.

class Participant(models.Model):
    # participant form fields
    EI = models.CharField(max_length=100)
    GA = models.CharField(max_length=100)

    L0 = models.CharField(max_length=100)
    L1 = models.CharField(max_length=100)
    L2 = models.CharField(max_length=100)
    L3 = models.CharField(max_length=100)

    LP1 = models.CharField(max_length=100)
    LP2 = models.CharField(max_length=100)

    LP1_load = models.CharField(max_length=100)
    LP2_load = models.CharField(max_length=100)




    class Meta:
        db_table = "participants"