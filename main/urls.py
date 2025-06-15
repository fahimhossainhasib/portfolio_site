from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_view, name="home"),
    path('blog/<slug:slug>/', views.blog_detail, name='blog_detail'),
    path('blog/', views.blog_list, name='blog_list')
]
