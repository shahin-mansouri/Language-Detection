from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    # صفحه اصلی تشخیص زبان
    path('', views.index, name='index'),
    
    # API endpoint
    path('api/detect/', views.detect_api, name='detect_api'),
    
    # تاریخچه تشخیص‌ها (نیاز به ورود)
    path('history/', views.history, name='history'),
    
    # آمار و گزارش‌ها
    path('statistics/', views.statistics, name='statistics'),
    
    # زبان‌های پشتیبانی شده
    path('languages/', views.supported_languages, name='supported_languages'),
    
    # بازخورد کاربران
    path('feedback/<int:detection_id>/', views.feedback, name='feedback'),
    
    # آموزش مدل (فقط ادمین)
    path('train/', views.train_model, name='train_model'),
]