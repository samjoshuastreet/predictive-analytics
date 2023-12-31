# Generated by Django 4.2.7 on 2023-12-16 09:10

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('myapp', '0002_delete_datasets'),
    ]

    operations = [
        migrations.CreateModel(
            name='Datasets',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=64)),
                ('dataset', models.FileField(upload_to='datasets/')),
                ('dataset_type', models.CharField(choices=[('Classification', 'Classification'), ('Regression', 'Regression')], max_length=15)),
            ],
        ),
    ]
