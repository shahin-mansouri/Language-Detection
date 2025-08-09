## Language-Detection (Django + scikit-learn)

سامانه وب «تشخیص زبان متن» با جنگو که با استفاده از الگوریتم Naive Bayes و بردار‌سازی Bag-of-Words (CountVectorizer) زبان متن ورودی را تشخیص می‌دهد. رابط وب کاربرپسند، تاریخچهٔ تشخیص‌ها، بازخورد کاربر، صفحهٔ آمار، و API مبتنی بر Django REST Framework را شامل می‌شود.

### امکانات کلیدی
- **تشخیص زبان متن به‌صورت آنی** در رابط وب: مسیر `detection/`
- **API عمومی** برای تشخیص زبان: مسیر `api/detect/` (DRF) و `detection/api/detect/` (ساده)
- **تاریخچهٔ تشخیص‌ها** برای کاربران وارد‌شده: `detection/history/`
- **ثبت بازخورد** برای هر تشخیص: `detection/feedback/<id>/`
- **صفحهٔ آمار و گزارش‌ها**: `detection/statistics/`
- **فهرست زبان‌های پشتیبانی‌شده**: `detection/languages/`
- **آموزش مجدد مدل (فقط ادمین)**: `detection/train/`
- **حساب کاربری**: ورود/خروج/ثبت‌نام/پروفایل در `accounts/`

### مسیرهای اصلی پروژه
- `admin/`: پنل مدیریت جنگو
- `api/`: APIهای مبتنی بر DRF
  - `api/detect/` (POST)
  - `api/languages/` (GET)
- `detection/`: رابط کاربری و برخی endpointها
  - `detection/` (صفحهٔ اصلی تشخیص)
  - `detection/api/detect/` (POST، معاف از CSRF)
  - `detection/history/`, `detection/statistics/`, `detection/languages/`, `detection/feedback/<id>/`, `detection/train/`
- `accounts/`: احراز هویت
  - `accounts/login/`, `accounts/logout/`, `accounts/register/`, `accounts/profile/`

## نصب و اجرا

### پیش‌نیازها
- Python 3.x
- pip و virtualenv (پیشنهادی)

### مراحل نصب
1) کلون کردن مخزن و ورود به پوشهٔ پروژه
```bash
git clone <repo-url>
cd Language-Detection
```

2) ساخت و فعال‌سازی محیط مجازی
```bash
python -m venv venv
venv\Scripts\activate  # ویندوز
# یا
source venv/bin/activate  # لینوکس/مک
```

3) نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

4) انجام مهاجرت‌های پایگاه‌داده
```bash
python manage.py migrate
```

5) اجرای سرور توسعه
```bash
python manage.py runserver
```

سپس به آدرس `http://127.0.0.1:8000/detection/` مراجعه کنید.

برای ساخت اکانت ادمین:
```bash
python manage.py createsuperuser
```

## API

دو Endpoint برای تشخیص موجود است. خروجی آن‌ها کمی متفاوت است؛ مورد DRF خروجی را به درصد تبدیل می‌کند.

### 1) DRF: POST `api/detect/`
درخواست (JSON):
```json
{
  "text": "سلام حال شما چطور است؟"
}
```

پاسخ نمونه (موارد درصدی):
```json
{
  "detected_language": "persian",
  "confidence": 97.3,
  "all_probabilities": {
    "persian": 97.3,
    "english": 1.2,
    "arabic": 1.0
  },
  "processing_time": 0.004,
  "text_length": 22,
  "word_count": 4
}
```

### 2) ساده (CSRF-Exempt): POST `detection/api/detect/`
درخواست (JSON):
```json
{
  "text": "Bonjour comment allez vous"
}
```

پاسخ نمونه (مقادیر احتمال بین 0 و 1):
```json
{
  "success": true,
  "detected_language": "French",
  "language_code": "french",
  "confidence": 0.972,
  "all_probabilities": {
    "french": 0.972,
    "english": 0.012
  },
  "processing_time": 0.003,
  "text_length": 30,
  "word_count": 4
}
```

### فهرست زبان‌های فعال: GET `api/languages/`
```json
[
  { "code": "persian", "name": "Persian", "native_name": "فارسی" },
  { "code": "english", "name": "English", "native_name": "English" }
]
```

نکته: جدول `Language` به‌صورت خودکار در اولین تشخیص، برای کد زبان‌های جدید ایجاد/به‌روز می‌شود.

## زبان‌های پشتیبانی‌شده (پیش‌فرض مدل)
مدل به‌صورت پیش‌فرض برای زبان‌های زیر دادهٔ آموزشی داخلی دارد:

- english (English)
- persian (فارسی)
- arabic (العربية)
- french (Français)
- german (Deutsch)
- spanish (Español)
- italian (Italiano)
- russian (Русский)
- turkish (Türkçe)
- chinese (中文)
- japanese (日本語)
- hindi (हिन्दी)

کدهای بالا همان مقادیر `language_code` در خروجی API هستند.

## آموزش و مدیریت مدل
فایل‌های مدل در مسیر `detection/models/` ذخیره می‌شوند:
- `language_model.pkl`
- `vectorizer.pkl`
- `languages.pkl`

راه‌های آموزش:
- از طریق رابط وب (فقط ادمین): `detection/train/`
- از طریق کد (Django shell):
```bash
python manage.py shell
```
```python
from detection.language_detector import LanguageDetector
detector = LanguageDetector()
accuracy = detector.train_model()
print(accuracy)
```

پس از آموزش، مدل و بردارساز ذخیره می‌شوند و در فراخوانی‌های بعدی به‌صورت خودکار بارگذاری می‌گردند.

## ساختار داده‌ها (مدل‌ها)
- `Language`: اطلاعات زبان‌ها (`name`, `code`, `native_name`, `is_active`)
- `DetectionHistory`: تاریخچهٔ تشخیص‌ها به‌همراه `confidence_score`, `processing_time`, `text_length`, `word_count` و اطلاعات کاربر/IP
- `UserFeedback`: بازخورد کاربران برای هر تشخیص (صحیح/نادرست/نسبتاً صحیح) + زبان صحیح و توضیح
- `ModelTrainingLog`: لاگ آموزش مدل‌ها و پارامترها
- `APIUsage`: ردیابی سادهٔ مصرف API (در صورت استفاده)

## نکات پایگاه‌داده و پیکربندی
- به‌صورت پیش‌فرض SQLite استفاده می‌شود (`db.sqlite3`).
- برای تغییر پایگاه‌داده به `PostgreSQL`/… تنظیمات را در `language_detection/settings.py` به‌روزرسانی کنید.

## نکات توسعه
- نسخه‌های کتابخانه‌ها در `requirements.txt` مشخص شده‌اند.
- کد تشخیص در `detection/language_detector.py` قرار دارد و شامل توابع:
  - `train_model`, `predict_with_confidence`, `save_model`, `load_model`, `get_supported_languages`
- فرم‌ها و اعتبارسنجی ورودی در `detection/forms.py` پیاده‌سازی شده‌اند.
- قالب‌ها در `detection/templates/detection/` قرار دارند.

## مجوز
—
