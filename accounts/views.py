from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django import forms
from django.contrib.auth.decorators import login_required

# Create your views here.

def register(request):
    class RegisterForm(forms.Form):
        username = forms.CharField(max_length=150, label='نام کاربری')
        password1 = forms.CharField(widget=forms.PasswordInput, label='رمز عبور')
        password2 = forms.CharField(widget=forms.PasswordInput, label='تکرار رمز عبور')

        def clean(self):
            cleaned_data = super().clean()
            if cleaned_data.get('password1') != cleaned_data.get('password2'):
                raise forms.ValidationError('رمز عبور و تکرار آن یکسان نیستند.')
            return cleaned_data

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            if User.objects.filter(username=username).exists():
                messages.error(request, 'این نام کاربری قبلاً ثبت شده است.')
            else:
                User.objects.create_user(username=username, password=password)
                messages.success(request, 'ثبت نام با موفقیت انجام شد. اکنون می‌توانید وارد شوید.')
                return redirect('accounts:login')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

@login_required
def profile(request):
    return render(request, 'accounts/profile.html', {'user': request.user})
