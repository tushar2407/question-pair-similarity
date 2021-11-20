from django.urls import path
from main.views import home, test

urlpatterns=[
    path('', home, name="home"),
    path('test/', test, name="test"),
]