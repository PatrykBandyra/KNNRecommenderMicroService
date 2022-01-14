from django.urls import path
from . import views
from . import utils


app_name = 'recommendations'
urlpatterns = [
    path('recommendations/simple/<int:product_id>/', views.get_simple_recommendation, name='simple_recommendation'),
    path('recommendations/advanced/<int:user_id>/', views.get_more_advanced_recommendation,
         name='more_advanced_recommendation'),
    path('recommendations/best/<int:user_id>/', views.get_best_recommendation,
         name='best_recommendation'),
]

utils.on_server_start()
