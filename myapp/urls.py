from django.urls import path
from . import views

urlpatterns = [
    path("test", views.test, name="test"),
    path("", views.home, name="home"),
    path("loginform", views.loginform, name="loginform"),
    path("register", views.registerform, name="registerform"),
    path("logoutUser", views.logoutUser, name="logoutUser"),
    path("dashboard", views.dashboard, name="dashboard"),
    path("getdataset?=<int:id>", views.getdataset, name="getdataset"),
    path("delete_dataset?=<int:id>", views.delete_dataset, name="delete_dataset"),
    path("getvisualization?=<int:id>", views.getvisualization, name="getvisualization"),
    path("dataset_upload", views.dataset_upload, name="dataset_upload"),
    path("model_upload", views.model_upload, name="model_upload"),
    path("delete_visualization?=<int:id>", views.delete_visualization, name="delete_visualization"),
    path("model_trainer", views.model_trainer, name="model_trainer"),
    path("get_model?=<int:id>", views.get_model, name="get_model"),
    path("delete_model?=<int:id>", views.delete_model, name="delete_model"),
]
