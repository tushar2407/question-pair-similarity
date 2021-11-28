from django.urls import path
from main.views import home, test

from django.conf import settings
from django.conf.urls.static import static  

urlpatterns=[
    path('', home, name="home"),
    path('test/', test, name="test"),
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) \
    + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)