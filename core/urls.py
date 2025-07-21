from django.urls import path, include

urlpatterns = [
    path('', include('stock_app.urls')),
]