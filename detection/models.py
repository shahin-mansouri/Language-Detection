from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Language(models.Model):
    """مدل برای ذخیره اطلاعات زبان‌های پشتیبانی شده"""
    
    name = models.CharField(max_length=50, unique=True, verbose_name="نام زبان")
    code = models.CharField(max_length=10, unique=True, verbose_name="کد زبان")
    native_name = models.CharField(max_length=100, verbose_name="نام بومی", blank=True)
    is_active = models.BooleanField(default=True, verbose_name="فعال")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاریخ ایجاد")
    
    class Meta:
        verbose_name = "زبان"
        verbose_name_plural = "زبان‌ها"
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.code})"


class DetectionHistory(models.Model):
    """مدل برای ذخیره تاریخچه تشخیص زبان"""
    
    user = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        verbose_name="کاربر"
    )
    input_text = models.TextField(verbose_name="متن ورودی")
    detected_language = models.ForeignKey(
        Language, 
        on_delete=models.CASCADE, 
        verbose_name="زبان تشخیص داده شده",
        null=True,
        blank=True
    )
    confidence_score = models.FloatField(verbose_name="میزان اطمینان")
    processing_time = models.FloatField(verbose_name="زمان پردازش (ثانیه)", null=True, blank=True)
    ip_address = models.GenericIPAddressField(verbose_name="آدرس IP", null=True, blank=True)
    user_agent = models.TextField(verbose_name="User Agent", blank=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاریخ تشخیص")
    
    # اطلاعات اضافی برای تحلیل
    text_length = models.IntegerField(verbose_name="طول متن", default=0)
    word_count = models.IntegerField(verbose_name="تعداد کلمات", default=0)
    
    class Meta:
        verbose_name = "تاریخچه تشخیص"
        verbose_name_plural = "تاریخچه تشخیص‌ها"
        ordering = ['-created_at']
    
    def __str__(self):
        text_preview = self.input_text[:50] + "..." if len(self.input_text) > 50 else self.input_text
        return f"{text_preview} -> {self.detected_language}"
    
    def save(self, *args, **kwargs):
        # محاسبه تعداد کلمات و طول متن
        if self.input_text:
            self.text_length = len(self.input_text)
            self.word_count = len(self.input_text.split())
        super().save(*args, **kwargs)


class ModelTrainingLog(models.Model):
    """مدل برای ذخیره لاگ آموزش مدل"""
    
    version = models.CharField(max_length=20, verbose_name="نسخه مدل")
    accuracy = models.FloatField(verbose_name="دقت مدل")
    training_samples = models.IntegerField(verbose_name="تعداد نمونه‌های آموزشی")
    training_time = models.FloatField(verbose_name="زمان آموزش (ثانیه)")
    parameters = models.JSONField(verbose_name="پارامترهای مدل", default=dict)
    notes = models.TextField(verbose_name="یادداشت‌ها", blank=True)
    is_active = models.BooleanField(default=False, verbose_name="مدل فعال")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاریخ آموزش")
    created_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        verbose_name="آموزش دهنده"
    )
    
    class Meta:
        verbose_name = "لاگ آموزش مدل"
        verbose_name_plural = "لاگ‌های آموزش مدل"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"مدل {self.version} - دقت: {self.accuracy:.2%}"
    
    def save(self, *args, **kwargs):
        # اگر این مدل فعال شود، بقیه را غیرفعال کن
        if self.is_active:
            ModelTrainingLog.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)


class UserFeedback(models.Model):
    """مدل برای ذخیره بازخورد کاربران"""
    
    FEEDBACK_CHOICES = [
        ('correct', 'صحیح'),
        ('incorrect', 'نادرست'),
        ('partially_correct', 'نسبتاً صحیح'),
    ]
    
    detection_history = models.OneToOneField(
        DetectionHistory, 
        on_delete=models.CASCADE, 
        verbose_name="تاریخچه تشخیص"
    )
    feedback = models.CharField(
        max_length=20, 
        choices=FEEDBACK_CHOICES, 
        verbose_name="بازخورد"
    )
    correct_language = models.ForeignKey(
        Language, 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True,
        verbose_name="زبان صحیح"
    )
    comment = models.TextField(verbose_name="نظر", blank=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاریخ بازخورد")
    
    class Meta:
        verbose_name = "بازخورد کاربر"
        verbose_name_plural = "بازخوردهای کاربران"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"بازخورد: {self.get_feedback_display()}"


class APIUsage(models.Model):
    """مدل برای ردیابی استفاده از API"""
    
    api_key = models.CharField(max_length=100, verbose_name="کلید API", null=True, blank=True)
    endpoint = models.CharField(max_length=100, verbose_name="نقطه پایانی")
    method = models.CharField(max_length=10, verbose_name="متد HTTP")
    ip_address = models.GenericIPAddressField(verbose_name="آدرس IP")
    user_agent = models.TextField(verbose_name="User Agent", blank=True)
    response_time = models.FloatField(verbose_name="زمان پاسخ (میلی‌ثانیه)")
    status_code = models.IntegerField(verbose_name="کد وضعیت")
    request_size = models.IntegerField(verbose_name="اندازه درخواست (بایت)", default=0)
    response_size = models.IntegerField(verbose_name="اندازه پاسخ (بایت)", default=0)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="زمان درخواست")
    
    class Meta:
        verbose_name = "استفاده از API"
        verbose_name_plural = "استفاده‌های API"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.method} {self.endpoint} - {self.status_code}"