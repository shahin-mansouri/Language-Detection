from django import forms
from .models import Language, UserFeedback

class DetectionForm(forms.Form):
    """فرم تشخیص زبان"""
    
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'متن خود را برای تشخیص زبان وارد کنید...',
            'rows': 5,
            'dir': 'auto'
        }),
        label='متن ورودی',
        max_length=5000,
        help_text='حداکثر 5000 کاراکتر'
    )
    
    def clean_text(self):
        text = self.cleaned_data.get('text', '').strip()
        
        if not text:
            raise forms.ValidationError('لطفاً متنی وارد کنید.')
        
        if len(text) < 3:
            raise forms.ValidationError('متن باید حداقل 3 کاراکتر باشد.')
        
        if len(text) > 5000:
            raise forms.ValidationError('متن نباید بیش از 5000 کاراکتر باشد.')
        
        return text


class FeedbackForm(forms.ModelForm):
    """فرم بازخورد کاربران"""
    
    class Meta:
        model = UserFeedback
        fields = ['feedback', 'correct_language', 'comment']
        widgets = {
            'feedback': forms.Select(attrs={'class': 'form-control'}),
            'correct_language': forms.Select(attrs={'class': 'form-control'}),
            'comment': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'نظر یا توضیح اضافی...'
            })
        }
        labels = {
            'feedback': 'نظر شما درباره دقت تشخیص',
            'correct_language': 'زبان صحیح (در صورت نادرست بودن تشخیص)',
            'comment': 'توضیحات اضافی'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['correct_language'].queryset = Language.objects.filter(is_active=True)
        self.fields['correct_language'].required = False
        self.fields['comment'].required = False


class BulkDetectionForm(forms.Form):
    """فرم تشخیص دسته‌ای زبان"""
    
    texts = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'هر متن را در یک خط جداگانه وارد کنید...',
            'rows': 10
        }),
        label='متون ورودی (هر خط یک متن)',
        help_text='هر متن را در یک خط جداگانه وارد کنید. حداکثر 50 متن'
    )
    
    def clean_texts(self):
        texts_input = self.cleaned_data.get('texts', '').strip()
        
        if not texts_input:
            raise forms.ValidationError('لطفاً متن‌هایی وارد کنید.')
        
        # تقسیم به خطوط
        lines = [line.strip() for line in texts_input.split('\n') if line.strip()]
        
        if not lines:
            raise forms.ValidationError('لطفاً متن‌های معتبری وارد کنید.')
        
        if len(lines) > 50:
            raise forms.ValidationError('حداکثر 50 متن قابل پردازش است.')
        
        # بررسی طول هر متن
        for i, line in enumerate(lines, 1):
            if len(line) < 3:
                raise forms.ValidationError(f'متن خط {i} باید حداقل 3 کاراکتر باشد.')
            if len(line) > 1000:
                raise forms.ValidationError(f'متن خط {i} نباید بیش از 1000 کاراکتر باشد.')
        
        return lines


class LanguageForm(forms.ModelForm):
    """فرم مدیریت زبان‌ها"""
    
    class Meta:
        model = Language
        fields = ['name', 'code', 'native_name', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'code': forms.TextInput(attrs={'class': 'form-control'}),
            'native_name': forms.TextInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'})
        }
        labels = {
            'name': 'نام زبان',
            'code': 'کد زبان',
            'native_name': 'نام بومی',
            'is_active': 'فعال'
        }
    
    def clean_code(self):
        code = self.cleaned_data.get('code', '').lower().strip()
        
        if not code:
            raise forms.ValidationError('کد زبان الزامی است.')
        
        # بررسی یکتا بودن کد (به جز خود رکورد در حالت ویرایش)
        queryset = Language.objects.filter(code=code)
        if self.instance.pk:
            queryset = queryset.exclude(pk=self.instance.pk)
        
        if queryset.exists():
            raise forms.ValidationError('این کد زبان قبلاً استفاده شده است.')
        
        return code