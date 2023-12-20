from django.db import models

# Create your models here.
class Datasets(models.Model):

    CLASSIFICATION = 'Classification'
    REGRESSION = 'Regression'

    DATASET_TYPE_CHOICES = [
        (CLASSIFICATION, 'Classification'),
        (REGRESSION, 'Regression'),
    ]

    filename = models.CharField(max_length=25)
    dataset = models.FileField(upload_to="datasets/")
    dataset_type = models.CharField(max_length=15, choices=DATASET_TYPE_CHOICES)

    def __str__(self):
        return self.filename
    
class Classification_Visualizations(models.Model):
    PIE_CHART = 'Pie Chart'
    HISTOGRAM = 'Histogram'

    CLASSIFICATION_VISUALIZATION_CHOICES = [
        (PIE_CHART, 'Pie Chart'),
        (HISTOGRAM, 'Histogram'),        
    ]

    visualization_type = models.CharField(max_length=9, choices=CLASSIFICATION_VISUALIZATION_CHOICES)
    dataset = models.IntegerField()
    title = models.CharField(max_length=255)
    target_column = models.CharField(max_length=255)

# class Regression_Visualizations(models.Model):
#     SCATTER_PLOT = 'Scatter Plot'
#     VIOLIN_PLOT = 'Violin Plot'

#     CLASSIFICATION_VISUALIZATION_CHOICES = [
#         (SCATTER_PLOT, 'Scatter Plot'),
#         (VIOLIN_PLOT, 'Violin Plot'),        
#     ]

#     visualization_type = models.CharField(max_length=12, choices=CLASSIFICATION_VISUALIZATION_CHOICES)
#     dataset = models.IntegerField()
#     title = models.CharField(max_length=255)
#     target_column = models.CharField(max_length=255),
#     target_column_two = models.CharField(max_length=255)

class Vizualization_for_Regression(models.Model):
    SCATTER_PLOT = 'Scatter Plot'
    VIOLIN_PLOT = 'Violin Plot'

    REGRESSION_VISUALIZATION_CHOICES = [
        (SCATTER_PLOT, 'Scatter Plot'),
        (VIOLIN_PLOT, 'Violin Plot'),        
    ]

    visualization_type = models.CharField(max_length=12, choices=REGRESSION_VISUALIZATION_CHOICES)
    dataset = models.IntegerField()
    title = models.CharField(max_length=255)
    target_column = models.CharField(max_length=255),
    target_column_two = models.CharField(max_length=255)
    target_column_one = models.CharField(max_length=255)

class Trained_Models(models.Model):

    filename = models.CharField(max_length=25)
    model = models.FileField(upload_to="models/")
    dataset = models.TextField()
    model_type = models.CharField(max_length=15)
    resampling_technique = models.IntegerField()
    algorithm = models.IntegerField()
    metric = models.IntegerField()

    def __str__(self):
        return self.filename
        
class New_Trained_Models(models.Model):

    filename = models.CharField(max_length=25)
    model = models.FileField(upload_to="models/")
    dataset_id = models.IntegerField()
    model_type = models.CharField(max_length=15)
    resampling_id = models.IntegerField()
    algorithm_id = models.IntegerField()
    metric_id = models.IntegerField()

    def __str__(self):
        return self.filename
