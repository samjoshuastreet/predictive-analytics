from django.shortcuts import render, redirect, HttpResponse, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from myapp.forms import DatasetForm
from myapp.models import Datasets, Classification_Visualizations, Vizualization_for_Regression, New_Trained_Models
import os
from django.http import JsonResponse
from django.apps import apps

# Create your views here.
def home(request):
    return render(request, "landing/base.html")

def loginform(request):

    if request.method == "POST":

        username = request.POST.get('username')
        pass1 = request.POST.get('pass')

        user = authenticate(request, username=username, password=pass1)
    
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.warning(request, 'Username or password is incorrect')
    
    return render(request, "landing/login.html")

def registerform(request):

    if request.method == 'POST':

        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if password1 != password2:
            messages.warning(request, 'Passwords do not match!')
        else:
            myUser = User.objects.create_user(username, email, password1)
            myUser.save()

        return render(request, "landing/login.html")
            
    return render(request, "landing/register.html")

def logoutUser(request):
    logout(request)
    return redirect('home')

def dashboard(request):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')
    datasets = Datasets.objects.all()
    models = New_Trained_Models.objects.all()
    context = {
        'classification_data': classification_data,
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'clasmod': Classification_Models,
        'regmod': Regression_Models,
        'datasets': datasets,
        'models': models
    }
    return render(request, 'dashboard/base.html', context)

import csv
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

def getdataset(request, id):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')
    datasets = Datasets.objects.all()
    models = New_Trained_Models.objects.all()

    dataset = Datasets.objects.get(id=id)
    csv_file_path = dataset.dataset.path
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        data = [row for row in csv_reader]

    page = request.GET.get('page', 1)
    paginator = Paginator(data, 15)

    try:
        data_page = paginator.page(page)
    except PageNotAnInteger:
        data_page = paginator.page(1)
    except EmptyPage:
        data_page = paginator.page(paginator.num_pages)


    context = {
        'classification_data': classification_data, 
        'regression_data': regression_data, 
        'dataset': dataset, 
        'header': header, 
        'data': data_page,
        'classvis': ClassVis,
        'regvis': RegVis,
        'clasmod': Classification_Models,
        'regmod': Regression_Models,
        'datasets': datasets,
        'models': models
    }

    if request.method == 'POST':
        visualization_type = request.POST.get('visualization_type')
        title = request.POST.get('title')
        target_dataset = request.POST.get('dataset')
        target_set = Datasets.objects.get(id=target_dataset)

        if target_set.dataset_type == 'Classification':
            
            target_column = request.POST.get('target_column')
        
            visualization = Classification_Visualizations.objects.create(
                visualization_type = visualization_type,
                title = title,
                dataset = target_dataset,
                target_column = target_column
            )
        else:
            target_column_one = request.POST.get('target_column_one')
            target_column_two = 'quality'

            visualization = Vizualization_for_Regression.objects.create(
                visualization_type = visualization_type,
                title = title,
                dataset = target_dataset,
                target_column_one = target_column_one,
                target_column_two = target_column_two
            )
        return render(request, "dashboard/getdataset.html", context)


    return render(request, 'dashboard/getdataset.html', context)

from django.core.files.storage import default_storage

def delete_dataset(request, id):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')
    datasets = Datasets.objects.all()
    models = New_Trained_Models.objects.all()

    target_dataset = Datasets.objects.get(id=id)
    file_path = target_dataset.dataset.path
    default_storage.delete(file_path)
    target_dataset.delete()

    mess = ''

    if target_dataset.dataset_type == Datasets.CLASSIFICATION:
        related_visualizations = Classification_Visualizations.objects.filter(dataset=id)
        related_visualizations.delete()
        try:
            related_models = New_Trained_Models.objects.filter(dataset_id=id, model_type='Classification')
            for x in related_models:
                file_path = x.model.path
                default_storage.delete(file_path)
            related_models.delete()
        except New_Trained_Models.DoesNotExist:
            mess = 'Meh'
    elif target_dataset.dataset_type == Datasets.REGRESSION:
        related_visualizations = Vizualization_for_Regression.objects.filter(dataset=id)
        related_visualizations.delete()
        try:
            related_models = New_Trained_Models.objects.filter(dataset_id=id, model_type='Regression')
            for x in related_models:
                file_path = x.model.path
                default_storage.delete(file_path)
            related_models.delete()
        except New_Trained_Models.DoesNotExist:
            mess = 'Meh'

    context = {
        'classification_data': classification_data,
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'clasmod': Classification_Models,
        'regmod': Regression_Models,
        'datasets': datasets,
        'models': models
    }
                
    messages.success(request, 'Dataset and associated files deleted successfully!')

    return render(request, "dashboard/base.html", context)

import plotly.express as px
import plotly.graph_objects as go
from django.http import Http404

def getvisualization(request, id):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')

    # Try to find the visualization instance in Classification_Visualizations
    try:
        target_visualization = get_object_or_404(Classification_Visualizations, id=id)
    except Http404:
        target_visualization = None


    # If not found, try finding it in Vizualization_for_Regression
    if target_visualization == None:
        target_visualization = get_object_or_404(Vizualization_for_Regression, id=id)

    target_dataset = Datasets.objects.get(id=target_visualization.dataset)

    csv_file_path = target_dataset.dataset.path
    target_column_values = []
    x_values = []
    y_values = []
    if target_dataset.dataset_type == 'Classification':
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Assuming the first row is the header
            target_column_index = header.index(target_visualization.target_column)

            for row in csv_reader:
                target_column_values.append(row[target_column_index])
    else:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Assuming the first row is the header
            x_values_index = header.index(target_visualization.target_column_one)
            y_values_index = header.index(target_visualization.target_column_two)

            for row in csv_reader:
                x_values.append(row[x_values_index])
                y_values.append(row[y_values_index])
    chart = ''

    if target_visualization.visualization_type == '1':
        # Histogram
        column_data = target_column_values

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=column_data, texttemplate="%{x}", textfont_size=20))
        fig.update_layout(
            width=1000,  
            height=600,  
            title=target_visualization.title,
            xaxis_title=target_visualization.target_column,
            yaxis_title="Count" 
        )
        chart = fig.to_html()
    elif target_visualization.visualization_type == '2':
        # Bar Chart
        fig = go.Figure(data=[go.Bar(x=list(range(1, len(target_column_values) + 1)), y=target_column_values)])

        fig.update_layout(
            width=1000,  
            height=600, 
            title=target_visualization.title,
            xaxis=dict(title='Count'),
            yaxis=dict(title=target_visualization.target_column),
        )
        chart = fig.to_html
    elif target_visualization.visualization_type == '3':
        # Scatter Plot
        fig = px.scatter(x=x_values, y=y_values)
        fig.update_layout(
            width=1000,  
            height=600,  
            title="Scatter plot for " + target_visualization.target_column_one + " and " + target_visualization.target_column_two + "." , 
            xaxis_title=target_visualization.target_column_one,  
            yaxis_title=target_visualization.target_column_two 
        )
        chart = fig.to_html
    elif target_visualization.visualization_type == '4':
        # Violin Plot
        fig = go.Figure()
        fig.add_trace(go.Violin(y=x_values, box_visible=True, line_color='blue'))
        fig.update_layout(
            width=1000,  
            height=600, 
            title=(target_visualization.title),
            yaxis_title=target_visualization.target_column_one,
            showlegend=True
        )
        chart = fig.to_html()

    context = { 
        'classification_data': classification_data, 
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'chart': chart,
        'title': target_visualization.title,
        'target_vis': target_visualization,
        'clasmod': Classification_Models,
        'regmod': Regression_Models
    }

    return render(request, 'dashboard/getvisualization.html', context)

def delete_visualization(request, id):

    # Try to find the visualization instance in Classification_Visualizations
    try:
        target_visualization = get_object_or_404(Classification_Visualizations, id=id)
    except Http404:
        target_visualization = None

    if target_visualization == None:
        target_visualization = get_object_or_404(Vizualization_for_Regression, id=id)

    target_visualization.delete()

    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')
    datasets = Datasets.objects.all()
    models = New_Trained_Models.objects.all()
    context = {
        'classification_data': classification_data,
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'clasmod': Classification_Models,
        'regmod': Regression_Models,
        'datasets': datasets,
        'models': models
    }
    return render(request, 'dashboard/base.html', context)

def import_test(request):
    source = "C:\\Users\\User\\Downloads\\MSU-IIT NMPC Rectangles (500 x 100 px).png"
    destination = "C:\\Users\\User\\Desktop\\ITD105 CASE STUDY\\new_case_3\\newcase\\myapp\\saved_models\\moved.png"

    try:
        if os.path.exists(destination):
            print("There is already a file there!")
        else:
            os.replace(source, destination)
            print(source + " was moved!")
    except FileNotFoundError:
        print(source+" was not found")


    return HttpResponse("Hello")

def dataset_upload(request):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')

    context = {
        'form': DatasetForm(), 
        'classification_data': classification_data, 
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'clasmod': Classification_Models,
        'regmod': Regression_Models
    }

    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your dataset was uploaded successfully!')
        else:
            context = {'form': form}
            return render(request, "dashboard/dataset_upload.html", context)

    return render(request, "dashboard/dataset_upload.html", context)

def model_upload(request):
    if request.method == 'POST':
        source = request.POST.get('model_source')
        print(source)
    
    return render(request, "dashboard/model_upload.html")

def test(request):
    field_names = [field.name for field in Vizualization_for_Regression._meta.get_fields()]
    print("Field Names:", field_names)

import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from django.conf import settings
import pickle
from urllib.parse import quote
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

def get_metrics(id):
    id = str(id)
    if id == '1':
        return 'Classification Accuracy'
    elif id == '2':
        return 'R-squared'
    
def get_ml(id):
    id = str(id)
    if id == '1':
        return 'Classification and Regression Trees (CART)'
    elif id == '2':
        return 'Random Forest'
    
def get_resampling(id):
    id = str(id)
    if id == '1':
        return 'Split Into Train & Test Sets'
    elif id == '2':
        return 'K-fold Cross Validation'

def performCART(parameters):
    target_dataset_path = parameters['target_dataset_path']
    target_dataset = parameters['target_dataset']
    resampling_technique_id = parameters['resampling_technique_id']
    test_size = parameters['test_size']
    folds = parameters['folds']
    max_depth = parameters['max_depth']
    performance_metric_id = parameters['performance_metric_id']
    action = parameters['action']

    if target_dataset.dataset_type == "Classification":
        # Loading the dataset
        dataframe = read_csv(target_dataset_path)
        X = dataframe.copy()
        column_names = dataframe.columns
        last_column_name = column_names[-1]
        y = dataframe[last_column_name]
        X = X.drop(columns=[last_column_name])

        # Resampling and Training
        if resampling_technique_id == '1':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            clf = DecisionTreeClassifier(ccp_alpha=0.01, max_depth=3)
            clf = clf.fit(X_train, y_train)

            # Testing
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracy = f'{accuracy * 100:.2f}%'

            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy,
                'model_path': model_path
            }
            return result
        elif resampling_technique_id == '2':
            clf = DecisionTreeClassifier(max_depth=max_depth)
            clf = clf.fit(X, y)
            cv_scores = cross_val_score(clf, X, y, cv=folds)
            accuracy_list = []

            # Testing
            for fold, accuracy in enumerate(cv_scores, 1):
                accuracy_list.append(f'Fold {fold}: Accuracy: {accuracy * 100:.2f}%')

            average_accuracy = cv_scores.mean()
            accuracy_list.append(f'Average Cross-Validated Accuracy: {average_accuracy * 100:.2f}%')

            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy_list,
                'model_path': model_path
            }
            return result
    else:
        # Loading the dataset
        dataframe = read_csv(target_dataset_path)
        X = dataframe.copy()
        column_names = dataframe.columns
        last_column_name = column_names[-1]
        y = dataframe[last_column_name]
        X = X.drop(columns=[last_column_name])

        # Resampling and Training
        if resampling_technique_id == '1':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            clf = DecisionTreeRegressor(max_depth=max_depth)
            clf = clf.fit(X_train, y_train)

            # Testing
            predictions = clf.predict(X_test)
            r2 = r2_score(y_test, predictions)
            accuracy = f'R-squared: {r2:.2f}'
            
            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy,
                'model_path': model_path
            }

            return result
        elif resampling_technique_id == '2':
            clf = DecisionTreeRegressor(max_depth=max_depth)
            clf = clf.fit(X, y)
            cv_scores = cross_val_score(clf, X, y, cv=folds, scoring='r2')
            accuracy_list = []

            # Testing
            for fold, r2 in enumerate(cv_scores, 1):
                accuracy_list.append(f'Fold {fold}: R-squared: {r2:2f}')
            
            average_accuracy = cv_scores.mean()
            accuracy_list.append(f'Average Cross-Validated R-squared: {average_accuracy:.2f}')

            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy_list,
                'model_path': model_path
            }
            return result
        
def performRF(parameters):
    target_dataset_path = parameters['target_dataset_path']
    target_dataset = parameters['target_dataset']
    resampling_technique_id = parameters['resampling_technique_id']
    test_size = parameters['test_size']
    folds = parameters['folds']
    max_depth = parameters['max_depth']
    performance_metric_id = parameters['performance_metric_id']
    action = parameters['action']

    if target_dataset.dataset_type == "Classification":
        # Loading the dataset
        dataframe = read_csv(target_dataset_path)
        X = dataframe.copy()
        column_names = dataframe.columns
        last_column_name = column_names[-1]
        y = dataframe[last_column_name]
        X = X.drop(columns=[last_column_name])

        # Resampling and Training
        if resampling_technique_id == '1':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            clf = RandomForestClassifier(max_depth=max_depth)
            clf = clf.fit(X_train, y_train)

            # Testing
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracy = f'{accuracy * 100:.2f}%'

            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy,
                'model_path': model_path
            }
            return result   
        elif resampling_technique_id == '2':
            clf = RandomForestClassifier(max_depth=max_depth)
            clf = clf.fit(X, y)
            cv_scores = cross_val_score(clf, X, y, cv=folds)
            accuracy_list = []

            # Testing
            for fold, accuracy in enumerate(cv_scores, 1):
                accuracy_list.append(f'Fold {fold}: Accuracy: {accuracy * 100:.2f}%')
            average_accuracy = cv_scores.mean()
            accuracy_list.append(f'Average Cross-Validated Accuracy: {average_accuracy * 100:.2f}%')

            # Create a cross-validation object (KFold)
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)

            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy_list,
                'model_path': model_path
            }
            return result
    else:
        # Loading the dataset
        dataframe = read_csv(target_dataset_path)
        X = dataframe.copy()
        column_names = dataframe.columns
        last_column_name = column_names[-1]
        y = dataframe[last_column_name]
        X = X.drop(columns=[last_column_name])

        # Resampling and Training
        if resampling_technique_id == '1':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            clf = RandomForestRegressor(max_depth=max_depth)
            clf = clf.fit(X_train, y_train)

            # Testing
            predictions = clf.predict(X_test)
            r2 = r2_score(y_test, predictions)
            accuracy = f'R-squared: {r2:.2f}'

            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy,
                'model_path': model_path
            }

            return result
        elif resampling_technique_id == '2':
            clf = RandomForestRegressor(max_depth=max_depth)
            clf = clf.fit(X, y)
            cv_scores = cross_val_score(clf, X, y, cv=folds, scoring='r2')
            accuracy_list = []

            # Testing
            for fold, r2 in enumerate(cv_scores, 1):
                accuracy_list.append(f'Fold {fold}: R-squared: {r2:.2f}')
            
            average_accuracy = cv_scores.mean()
            accuracy_list.append(f'Average Cross-Validated R-squared: {average_accuracy:.2f}')

            # Saving Model
            model_filename = f'{target_dataset.filename}.joblib'
            model_folder = os.path.join(settings.MEDIA_ROOT, 'savedmodels')

            # Check for existing files with similar names
            existing_models = [f for f in os.listdir(model_folder) if f.startswith(model_filename)]

            # If there are existing models, find the next available number
            model_number = 1
            while f'{model_filename}_{model_number}.joblib' in existing_models:
                model_number += 1

            # Create the final model filename with the appended number
            final_model_filename = f'{model_filename}_{model_number}.joblib'
            model_path = os.path.join(model_folder, final_model_filename)

            # Save the model
            with open(model_path, 'wb') as file:
                pickle.dump(clf, file)

            result = {
                'result': accuracy_list,
                'model_path': model_path
            }
            return result

def model_trainer(request):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')
    datasets = Datasets.objects.all()
    context = {
        'classification_data': classification_data, 
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'datasets': datasets,
        'clasmod': Classification_Models,
        'regmod': Regression_Models
    }


    if request.method == 'POST':

        target_dataset_id = request.POST.get('target_dataset')
        target_dataset = Datasets.objects.get(id=target_dataset_id)
        target_dataset_path = target_dataset.dataset

        resampling_technique_id = request.POST.get('resampling_technique')
        test_size = request.POST.get('test_size') # Split Train & Test
        if test_size == '':
            test_size = 0.20
        test_size = float(test_size)
        folds = request.POST.get('folds') # K-fold
        if folds == '':
            folds = 5
        folds = int(folds)



        ml_algorithm_id = request.POST.get('ml_algorithm')
        max_depth = request.POST.get('max_depth') # max_depth
        if max_depth == '':
            max_depth = 0
        max_depth = int(folds)

        performance_metric_id = request.POST.get('performance_metric')

        action = 'train'

        if ml_algorithm_id == '1':
            # CART
            parameters = {
                'target_dataset_path': target_dataset_path,
                'target_dataset': target_dataset,

                'resampling_technique_id': resampling_technique_id,
                'test_size': test_size,
                'folds': folds,
                
                'max_depth': max_depth,

                'performance_metric_id': performance_metric_id,
                
                'action': action
            }
            results = performCART(parameters)
        elif ml_algorithm_id == '2':
            # Random Forest
            parameters = {
                'target_dataset_path': target_dataset_path,
                'target_dataset': target_dataset,

                'resampling_technique_id': resampling_technique_id,
                'test_size': test_size,
                'folds': folds,
                
                'max_depth': max_depth,

                'performance_metric_id': performance_metric_id,

                'action': action
            }
            results = performRF(parameters)

        local_path = results['model_path']
        relative_path = os.path.relpath(local_path, settings.MEDIA_ROOT).replace("\\", "/")

        model = New_Trained_Models.objects.create(
            filename = target_dataset.filename,
            model = local_path,
            dataset_id = target_dataset_id,
            model_type = target_dataset.dataset_type,
            resampling_id = resampling_technique_id,
            algorithm_id = ml_algorithm_id,
            metric_id = resampling_technique_id
        )

        messages.success(request, 'Your model was uploaded successfully! Scroll down to view results!')

        context = {
            'classification_data': classification_data, 
            'regression_data': regression_data,
            'classvis': ClassVis,
            'regvis': RegVis,
            'datasets': datasets,
            'selected_model': target_dataset,
            'resampling_technique_id': resampling_technique_id,
            'resampling_technique': get_resampling(resampling_technique_id),
            'machine_learning': get_ml(ml_algorithm_id),
            'performance_metrics': get_metrics(performance_metric_id),
            'results': results,
            'relative_path': relative_path,
            'target_dataset_path': target_dataset_path,
            'clasmod': Classification_Models,
            'regmod': Regression_Models,
            'model': model
        }
        
        return render(request, "dashboard/model_trainer.html", context)
        
    return render(request, "dashboard/model_trainer.html", context)

import joblib
import numpy as np

def get_model(request, id):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')

    query = New_Trained_Models.objects.filter(id=id)
    target_model = query[0]
    target_dataset = Datasets.objects.get(id=target_model.dataset_id)
    resampling = get_resampling(target_model.resampling_id)
    algorithm = get_ml(target_model.algorithm_id)
    metric = get_metrics(target_model.metric_id)

    target_dataset_path = target_dataset.dataset.path
    csv_file_path = target_dataset_path

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        data = [row for row in csv_reader]
    
    df = pd.DataFrame(data, columns=header)

    # Get the minimum and maximum values for each column
    min_values = df.min()
    max_values = df.max()

    # Print the results
    print("Minimum values:")
    print(min_values)

    print("\nMaximum values:")
    print(max_values)
    
    columns_length = len(header)

    model = target_model.model

    context = {
        'classification_data': classification_data,
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'clasmod': Classification_Models,
        'regmod': Regression_Models,
        'target': target_model,
        'target_dataset': target_dataset,
        'resampling': resampling,
        'algorithm': algorithm,
        'metric': metric,
        'header': header,
        'column_length': columns_length,
    }

    if request.method == 'POST':
    
        prediction_data = []
        for key in request.POST:
            prediction_data.append(request.POST[key])
        

        prediction_data = prediction_data[1:]

        model = joblib.load(model)
        prediction_data_array = np.array(prediction_data).reshape(1, -1)

        y_pred = model.predict(prediction_data_array)

        context = {
            'classification_data': classification_data,
            'regression_data': regression_data,
            'classvis': ClassVis,
            'regvis': RegVis,
            'clasmod': Classification_Models,
            'regmod': Regression_Models,
            'target': target_model,
            'target_dataset': target_dataset,
            'resampling': resampling,
            'algorithm': algorithm,
            'metric': metric,
            'header': header,
            'column_length': columns_length,
            'results': y_pred[0]
        }
    return render(request, 'dashboard/get_model.html', context)

def delete_model(request, id):
    classification_data = Datasets.objects.filter(dataset_type='Classification')
    regression_data = Datasets.objects.filter(dataset_type='Regression')
    ClassVis = Classification_Visualizations.objects.all()
    RegVis = Vizualization_for_Regression.objects.all()
    Classification_Models = New_Trained_Models.objects.filter(model_type='Classification')
    Regression_Models = New_Trained_Models.objects.filter(model_type='Regression')
    datasets = Datasets.objects.all()
    models = New_Trained_Models.objects.all()

    target_model = New_Trained_Models.objects.get(id=id)
    file_path = target_model.model.path
    default_storage.delete(file_path)
    target_model.delete()

    context = {
        'classification_data': classification_data,
        'regression_data': regression_data,
        'classvis': ClassVis,
        'regvis': RegVis,
        'clasmod': Classification_Models,
        'regmod': Regression_Models,
        'datasets': datasets,
        'models': models
    }
                
    messages.success(request, 'Model deleted successfully!')

    return render(request, "dashboard/base.html", context)