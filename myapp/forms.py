from django import forms
from myapp.models import Datasets, Classification_Visualizations
from django.core.exceptions import ValidationError

def validate_csv_file(value):
    if not value.name.lower().endswith('.csv'):
        raise ValidationError('Only CSV files are allowed.')

class DatasetForm(forms.ModelForm):
    filename = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    dataset = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control', 'id': 'formFile'}),
        validators=[validate_csv_file]
    )
    dataset_type = forms.ChoiceField(
        choices=Datasets.DATASET_TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )


    class Meta:
        model = Datasets
        fields = ('filename', 'dataset', 'dataset_type')
