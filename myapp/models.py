from django.db import models

# Create your models here.

class Calculation(models.Model):
    # participant form fields
    EI = models.CharField(max_length=100, default='d')
    GA = models.CharField(max_length=100, default='d')

    L = models.CharField(max_length=100, default='d')

    LP1 = models.CharField(max_length=100, default='d')
    LP2 = models.CharField(max_length=100, default='d')

    q_load = models.IntegerField(blank=True, null=True, default=100)




    class Meta:
        db_table = "participants"