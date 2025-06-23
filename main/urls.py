from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_view, name="home"),
    path('blog/<slug:slug>/', views.blog_detail, name='blog_detail'),
    path('blog/', views.blog_list, name='blog_list'),
    path('projects/<slug:slug>/', views.project_detail, name='project_detail'),
    path('projects/<slug:slug>/demo/', views.project_demo, name='project_demo'),
    path('projects/', views.project_list, name='project_list'),
    path('clipsniper_demo/', views.clipsniper_demo, name='clipsniper_demo')
]
