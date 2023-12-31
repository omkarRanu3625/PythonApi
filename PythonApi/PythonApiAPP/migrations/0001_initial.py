# Generated by Django 4.2.6 on 2023-10-19 06:00

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PythonApiModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('DomainUUID', models.CharField(max_length=100, unique=True)),
                ('wav_file', models.FileField(upload_to='uploads/')),
                ('domain_name', models.CharField(max_length=100)),
            ],
        ),
    ]
