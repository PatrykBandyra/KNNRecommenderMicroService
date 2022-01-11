from django.urls import path
from . import views
from . import utils


app_name = 'recommendations'
urlpatterns = [
    path('recommendations/<int:product_id>/', views.get_simple_recommendation, name='simple_recommendation')
]
