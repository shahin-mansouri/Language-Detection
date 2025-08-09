from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Avg
from django.utils import timezone
import json
import time

from .language_detector import LanguageDetector
from .models import DetectionHistory, Language, UserFeedback
from .forms import DetectionForm, FeedbackForm


def get_client_ip(request):
    """دریافت IP کاربر"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def index(request):
    """صفحه اصلی تشخیص زبان"""
    form = DetectionForm()
    result = None
    
    if request.method == 'POST':
        form = DetectionForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            
            try:
                # تشخیص زبان
                detector = LanguageDetector()
                start_time = time.time()
                prediction_result = detector.predict_with_confidence(text)
                processing_time = time.time() - start_time
                
                if prediction_result['predicted_language']:
                    # پیدا کردن یا ایجاد زبان
                    language, created = Language.objects.get_or_create(
                        code=prediction_result['predicted_language'],
                        defaults={
                            'name': prediction_result['predicted_language'].title(),
                            'native_name': prediction_result['predicted_language'].title()
                        }
                    )
                    
                    # ذخیره در تاریخچه
                    detection_history = DetectionHistory.objects.create(
                        user=request.user if request.user.is_authenticated else None,
                        input_text=text,
                        detected_language=language,
                        confidence_score=prediction_result['confidence'],
                        processing_time=processing_time,
                        ip_address=get_client_ip(request),
                        user_agent=request.META.get('HTTP_USER_AGENT', ''),
                        text_length=prediction_result['text_length'],
                        word_count=len(text.split())
                    )
                    result = {
                        'detected_language': language.name,
                        'confidence': round(prediction_result['confidence']*100, 1),
                        'all_probabilities': {lang: round(prob * 100, 1) for lang, prob in prediction_result['all_probabilities'].items()},
                        'processing_time': processing_time,
                        'text_length': prediction_result['text_length'],
                        'word_count': len(text.split()),
                        'detection_id': detection_history.id
                    }
                    
                    messages.success(request, f'زبان متن با اطمینان {result["confidence"]:.1f}% تشخیص داده شد!')
                
                else:
                    messages.error(request, 'متن وارد شده قابل تشخیص نیست.')
                    
            except Exception as e:
                messages.error(request, f'خطا در تشخیص زبان: {str(e)}')
    
    # آمار کلی برای نمایش
    total_detections = DetectionHistory.objects.count()
    supported_languages = Language.objects.filter(is_active=True).count()
    
    context = {
        'form': form,
        'result': result,
        'total_detections': total_detections,
        'supported_languages': supported_languages,
    }
    
    return render(request, 'detection/index.html', context)


@require_http_methods(["POST"])
@csrf_exempt
def detect_api(request):
    """API endpoint برای تشخیص زبان"""
    try:
        # پارس کردن JSON
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        
        if not text:
            return JsonResponse({
                'error': 'متن ورودی خالی است',
                'success': False
            }, status=400)
        
        # تشخیص زبان
        detector = LanguageDetector()
        start_time = time.time()
        result = detector.predict_with_confidence(text)
        processing_time = time.time() - start_time
        
        if result['predicted_language']:
            # پیدا کردن یا ایجاد زبان
            language, created = Language.objects.get_or_create(
                code=result['predicted_language'],
                defaults={
                    'name': result['predicted_language'].title(),
                    'native_name': result['predicted_language'].title()
                }
            )
            
            # ذخیره در تاریخچه
            DetectionHistory.objects.create(
                input_text=text,
                detected_language=language,
                confidence_score=result['confidence'],
                processing_time=processing_time,
                ip_address=get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                text_length=result['text_length'],
                word_count=len(text.split())
            )
            
            return JsonResponse({
                'success': True,
                'detected_language': language.name,
                'language_code': language.code,
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities'],
                'processing_time': processing_time,
                'text_length': result['text_length'],
                'word_count': len(text.split())
            })
        
        else:
            return JsonResponse({
                'error': 'متن قابل تشخیص نیست',
                'success': False
            }, status=400)
    
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'فرمت JSON نامعتبر',
            'success': False
        }, status=400)
    
    except Exception as e:
        return JsonResponse({
            'error': f'خطا در پردازش: {str(e)}',
            'success': False
        }, status=500)


@login_required
def history(request):
    """نمایش تاریخچه تشخیص‌های کاربر"""
    detections = DetectionHistory.objects.filter(user=request.user).order_by('-created_at')
    
    # صفحه‌بندی
    paginator = Paginator(detections, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_detections': detections.count(),
    }
    
    return render(request, 'detection/history.html', context)


def statistics(request):
    """نمایش آمار عمومی"""
    # آمار کلی
    total_detections = DetectionHistory.objects.count()
    total_users = DetectionHistory.objects.values('user').distinct().count()
    
    # آمار زبان‌ها
    language_stats = DetectionHistory.objects.values(
        'detected_language__name', 'detected_language__code'
    ).annotate(
        count=Count('id'),
        avg_confidence=Avg('confidence_score')
    ).order_by('-count')[:10]
    
    # آمار روزانه (7 روز گذشته)
    from datetime import datetime, timedelta
    seven_days_ago = timezone.now() - timedelta(days=7)
    daily_stats = DetectionHistory.objects.filter(
        created_at__gte=seven_days_ago
    ).extra(
        select={'day': 'date(created_at)'}
    ).values('day').annotate(
        count=Count('id')
    ).order_by('day')
    
    # میانگین زمان پردازش
    avg_processing_time = DetectionHistory.objects.aggregate(
        avg_time=Avg('processing_time')
    )['avg_time'] or 0
    
    context = {
        'total_detections': total_detections,
        'total_users': total_users,
        'language_stats': language_stats,
        'daily_stats': daily_stats,
        'avg_processing_time': avg_processing_time,
        'supported_languages': Language.objects.filter(is_active=True).count(),
    }
    
    return render(request, 'detection/statistics.html', context)


@login_required
def feedback(request, detection_id):
    """ارسال بازخورد برای یک تشخیص"""
    try:
        detection = DetectionHistory.objects.get(id=detection_id, user=request.user)
    except DetectionHistory.DoesNotExist:
        messages.error(request, 'تشخیص مورد نظر یافت نشد.')
        return redirect('detection:history')
    
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            feedback_obj, created = UserFeedback.objects.get_or_create(
                detection_history=detection,
                defaults={
                    'feedback': form.cleaned_data['feedback'],
                    'correct_language': form.cleaned_data.get('correct_language'),
                    'comment': form.cleaned_data.get('comment', '')
                }
            )
            
            if not created:
                # به‌روزرسانی بازخورد موجود
                feedback_obj.feedback = form.cleaned_data['feedback']
                feedback_obj.correct_language = form.cleaned_data.get('correct_language')
                feedback_obj.comment = form.cleaned_data.get('comment', '')
                feedback_obj.save()
            
            messages.success(request, 'بازخورد شما ثبت شد. متشکریم!')
            return redirect('detection:history')
    
    else:
        form = FeedbackForm()
    
    context = {
        'form': form,
        'detection': detection,
    }
    
    return render(request, 'detection/feedback.html', context)


def train_model(request):
    """آموزش مجدد مدل (فقط برای ادمین)"""
    if not request.user.is_superuser:
        messages.error(request, 'شما دسترسی لازم را ندارید.')
        return redirect('detection:index')
    
    if request.method == 'POST':
        try:
            detector = LanguageDetector()
            start_time = time.time()
            accuracy = detector.train_model()
            training_time = time.time() - start_time
            
            # ذخیره لاگ آموزش
            from .models import ModelTrainingLog
            ModelTrainingLog.objects.create(
                version=f"v{timezone.now().strftime('%Y%m%d_%H%M%S')}",
                accuracy=accuracy,
                training_samples=0,  # باید از داده‌های واقعی محاسبه شود
                training_time=training_time,
                parameters={
                    'algorithm': 'MultinomialNB',
                    'vectorizer': 'CountVectorizer',
                    'max_features': 5000
                },
                is_active=True,
                created_by=request.user
            )
            
            messages.success(request, f'مدل با موفقیت آموزش داده شد. دقت: {accuracy:.2%}')
            
        except Exception as e:
            messages.error(request, f'خطا در آموزش مدل: {str(e)}')
    
    return render(request, 'detection/train_model.html')


def supported_languages(request):
    """نمایش زبان‌های پشتیبانی شده"""
    languages = Language.objects.filter(is_active=True).order_by('name')
    
    # آمار هر زبان
    language_stats = {}
    for lang in languages:
        stats = DetectionHistory.objects.filter(detected_language=lang).aggregate(
            count=Count('id'),
            avg_confidence=Avg('confidence_score')
        )
        language_stats[lang.id] = {
            'count': stats['count'] or 0,
            'avg_confidence': stats['avg_confidence'] or 0
        }
    
    context = {
        'languages': languages,
        'language_stats': language_stats,
    }
    
    return render(request, 'detection/supported_languages.html', context)