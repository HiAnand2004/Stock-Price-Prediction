from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.core.validators import validate_email
from django.utils.translation import gettext_lazy as _
import re
from .models import User

class  RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

class SignUpForm(UserCreationForm):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'id': 'id_email',
            'autocomplete': 'email'
        }),
        required=True,
        label=_("Email"),
        help_text=_("Enter a valid email address you have access to")
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({
            'class': 'form-input',
            'id': 'id_password1'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-input',
            'id': 'id_password2'
        })


    def clean_email(self):
        email = self.cleaned_data.get('email').lower().strip()
        
        # Basic format validation
        try:
            validate_email(email)
        except ValidationError:
            raise ValidationError(_("Please enter a valid email address (e.g., user@example.com)"))
        
        # Check for common email format mistakes
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValidationError(_("Email format is invalid. Please check for typos."))
        
        # Check for disposable emails
        disposable_domains = [
            'tempmail', 'mailinator', 'guerrillamail', 
            '10minutemail', 'throwawaymail', 'fakeinbox'
        ]
        if any(domain in email for domain in disposable_domains):
            raise ValidationError(_("Disposable email addresses are not allowed. Please use a permanent email."))
        
        # Check for existing user
        if User.objects.filter(email=email).exists():
            raise ValidationError(_("This email is already registered. Did you mean to login?"))
        
        return email

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        try:
            validate_password(password1, self.instance)
        except ValidationError as error:
            self.add_error('password1', error)
        return password1

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get("password1")
        password2 = cleaned_data.get("password2")
        
        if password1 and password2 and password1 != password2:
            self.add_error('password2', "Passwords don't match")
        
        return cleaned_data
