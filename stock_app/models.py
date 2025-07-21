# import os
# import django

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_predictor_project.settings')
# django.setup()

# from stock_app.models import NewsHeadline  # Now you can import models

from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.utils.text import slugify


class NewsHeadline(models.Model):
    id = models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')
    title = models.CharField(max_length=200)
    date = models.DateTimeField()
    url = models.URLField(max_length=500, blank=True, null=True)  # <-- Add this line
    summary = models.TextField(blank=True, null=True)
    sentiment_score = models.FloatField(default=0.0)
    sentiment_label = models.CharField(blank=True, max_length=20, null=True)
    ticker = models.CharField(max_length=20, default='^NSEI')  # <-- Add this line

    class Meta:
        app_label = 'stock_app'



class UserManager(BaseUserManager):  # Add this class
    def create_superuser(self, email, full_name, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
            
        return self.create_user(email, full_name, password, **extra_fields)

    def create_user(self, email, full_name, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, full_name=full_name, **extra_fields)
        user.set_password(password)
        user.save()
        return user

class User(AbstractUser):
    # Custom fields
    email = models.EmailField(unique=True)
    # Make full_name optional as it might not be in the original signup form
    full_name = models.CharField(max_length=150, blank=True, null=True) 
    
    # Remove username requirements
    username = models.CharField(max_length=150, unique=True, blank=True)
    
    # Use email as login field
    USERNAME_FIELD = 'email'
    # Removed 'full_name' from REQUIRED_FIELDS since it's now optional
    REQUIRED_FIELDS = [] 

    objects = UserManager()

    def save(self, *args, **kwargs):
        if not self.username and self.full_name: # Generate username from full_name if available
            base_username = slugify(self.full_name.lower().replace(' ', ''))
            self.username = base_username
            # Handle duplicates
            counter = 1
            while User.objects.filter(username=self.username).exists():
                self.username = f"{base_username}{counter}"
                counter += 1
        elif not self.username: # If full_name is also not provided, generate a generic username
            self.username = slugify(self.email.split('@')[0]) # Use email prefix
            counter = 1
            while User.objects.filter(username=self.username).exists():
                self.username = f"{slugify(self.email.split('@')[0])}{counter}"
                counter += 1

        super().save(*args, **kwargs)

    def __str__(self):
        return self.email

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='stock_app_users',
        blank=True,
        help_text='The groups this user belongs to.'
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='stock_app_users',
        blank=True,
        help_text='Specific permissions for this user.'
    )

    class Meta:
        app_label = 'stock_app'


class StockPrice(models.Model):
    ticker = models.CharField(max_length=20)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('ticker', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.ticker} - {self.date}"

features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
