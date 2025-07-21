import pytest
from django.test import Client
from django.urls import reverse
from django.contrib.auth import get_user_model

@pytest.mark.django_db
def test_signup():
    client = Client()
    signup_url = reverse('signup')
    response = client.post(signup_url, {
        'email': 'newuser@example.com',
        'password1': 'newpass123',
        'password2': 'newpass123',
    })
    assert response.status_code == 302
    assert get_user_model().objects.filter(email='newuser@example.com').exists()

@pytest.mark.django_db
def test_login_logout():
    client = Client()
    user = get_user_model().objects.create_user(email='testuser@example.com', password='testpass123')
    login_url = reverse('login')
    dashboard_url = reverse('dashboard')

    login = client.login(email='testuser@example.com', password='testpass123')
    assert login

    response = client.get(dashboard_url)
    assert response.status_code == 200

    client.logout()
    response = client.get(dashboard_url)
    assert response.status_code == 302

@pytest.mark.django_db
def test_prediction_view_requires_login():
    client = Client()
    prediction_url = reverse('predict')
    response = client.get(prediction_url)
    assert response.status_code == 302

@pytest.mark.django_db
def test_prediction_view_logged_in():
    client = Client()
    user = get_user_model().objects.create_user(email='testuser2@example.com', password='testpass123')
    client.login(email='testuser2@example.com', password='testpass123')
    prediction_url = reverse('predict')
    response = client.get(prediction_url)
    assert response.status_code == 200
    assert 'predictions' in response.context

@pytest.mark.django_db
def test_visualize_view_requires_login():
    client = Client()
    visualize_url = reverse('visualize')
    response = client.get(visualize_url)
    assert response.status_code == 302

@pytest.mark.django_db
def test_visualize_view_logged_in():
    client = Client()
    user = get_user_model().objects.create_user(email='testuser3@example.com', password='testpass123')
    client.login(email='testuser3@example.com', password='testpass123')
    visualize_url = reverse('visualize')
    response = client.get(visualize_url)
    assert response.status_code == 200
    assert 'plot_url' in response.context
