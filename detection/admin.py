from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import Language, DetectionHistory, ModelTrainingLog, UserFeedback, APIUsage


@admin.register(Language)
class LanguageAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'native_name', 'is_active', 'created_at', 'detection_count']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'code', 'native_name']
    list_editable = ['is_active']
    ordering = ['name']
    
    def detection_count(self, obj):
        count = obj.detectionhistory_set.count()
        if count > 0:
            url = reverse('admin:detection_detectionhistory_changelist')
            return format_html('<a href="{}?detected_language__id={}">{} تشخیص</a>', url, obj.id, count)
        return '0 تشخیص'
    detection_count.short_description = 'تعداد تشخیص'


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ['text_preview', 'detected_language', 'confidence_display', 'user', 'created_at', 'processing_time_display']
    list_filter = ['detected_language', 'created_at', 'confidence_score']
    search_fields = ['input_text', 'user__username', 'ip_address']
    readonly_fields = ['created_at', 'processing_time', 'ip_address', 'user_agent']
    date_hierarchy = 'created_at'
    list_per_page = 50
    
    fieldsets = (
        ('اطلاعات اصلی', {
            'fields': ('user', 'input_text', 'detected_language', 'confidence_score')
        }),
        ('آمار متن', {
            'fields': ('text_length', 'word_count'),
            'classes': ('collapse',)
        }),
        ('اطلاعات فنی', {
            'fields': ('processing_time', 'ip_address', 'user_agent', 'created_at'),
            'classes': ('collapse',)
        })
    )
    
    def text_preview(self, obj):
        preview = obj.input_text[:50] + "..." if len(obj.input_text) > 50 else obj.input_text
        return format_html('<span title="{}">{}</span>', obj.input_text, preview)
    text_preview.short_description = 'متن'
    
    def confidence_display(self, obj):
        percentage = obj.confidence_score * 100
        if percentage >= 80:
            color = 'green'
        elif percentage >= 60:
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {};">{:.1f}%</span>', color, percentage)
    confidence_display.short_description = 'اطمینان'
    
    def processing_time_display(self, obj):
        if obj.processing_time:
            return f'{obj.processing_time:.3f}s'
        return '-'
    processing_time_display.short_description = 'زمان پردازش'


@admin.register(ModelTrainingLog)
class ModelTrainingLogAdmin(admin.ModelAdmin):
    list_display = ['version', 'accuracy_display', 'training_samples', 'training_time_display', 'is_active', 'created_at', 'created_by']
    list_filter = ['is_active', 'created_at']
    search_fields = ['version', 'notes']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('اطلاعات مدل', {
            'fields': ('version', 'accuracy', 'is_active')
        }),
        ('آمار آموزش', {
            'fields': ('training_samples', 'training_time', 'parameters')
        }),
        ('متادیتا', {
            'fields': ('notes', 'created_by', 'created_at'),
            'classes': ('collapse',)
        })
    )
    
    def accuracy_display(self, obj):
        percentage = obj.accuracy * 100
        if percentage >= 90:
            color = 'green'
        elif percentage >= 80:
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {}; font-weight: bold;">{:.2f}%</span>', color, percentage)
    accuracy_display.short_description = 'دقت'
    
    def training_time_display(self, obj):
        if obj.training_time < 60:
            return f'{obj.training_time:.1f}s'
        else:
            minutes = obj.training_time // 60
            seconds = obj.training_time % 60
            return f'{int(minutes)}m {seconds:.1f}s'
    training_time_display.short_description = 'زمان آموزش'


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['detection_preview', 'feedback_display', 'correct_language', 'created_at']
    list_filter = ['feedback', 'correct_language', 'created_at']
    search_fields = ['detection_history__input_text', 'comment']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    
    def detection_preview(self, obj):
        text = obj.detection_history.input_text
        preview = text[:30] + "..." if len(text) > 30 else text
        return preview
    detection_preview.short_description = 'متن تشخیص داده شده'
    
    def feedback_display(self, obj):
        colors = {
            'correct': 'green',
            'incorrect': 'red',
            'partially_correct': 'orange'
        }
        color = colors.get(obj.feedback, 'black')
        return format_html('<span style="color: {};">{}</span>', color, obj.get_feedback_display())
    feedback_display.short_description = 'بازخورد'


@admin.register(APIUsage)
class APIUsageAdmin(admin.ModelAdmin):
    list_display = ['endpoint', 'method', 'status_code_display', 'response_time_display', 'ip_address', 'created_at']
    list_filter = ['endpoint', 'method', 'status_code', 'created_at']
    search_fields = ['ip_address', 'user_agent', 'api_key']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    list_per_page = 100
    
    def status_code_display(self, obj):
        if 200 <= obj.status_code < 300:
            color = 'green'
        elif 400 <= obj.status_code < 500:
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {}; font-weight: bold;">{}</span>', color, obj.status_code)
    status_code_display.short_description = 'وضعیت'
    
    def response_time_display(self, obj):
        if obj.response_time < 100:
            color = 'green'
        elif obj.response_time < 500:
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {};">{:.1f}ms</span>', color, obj.response_time)
    response_time_display.short_description = 'زمان پاسخ'


# تنظیمات سایت ادمین
admin.site.site_header = "مدیریت سیستم تشخیص زبان"
admin.site.site_title = "تشخیص زبان"
admin.site.index_title = "پنل مدیریت"