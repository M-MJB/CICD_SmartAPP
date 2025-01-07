import pytest
from app import app
import requests
import pandas as pd
import datetime


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


API_KEY = 'your_weatherapi_key_here'
BASE_URL = 'http://api.weatherapi.com/v1/current.json'

city ='london'
params = {
    'key': API_KEY,
    'q': city
}
response = requests.get(BASE_URL, params=params)
data = response.json()
print(data)

current = data['current']
weather_data = {
        'Temp_C': current['temp_c'],  # Temperature in Celsius
        'Press_kPa': current['pressure_mb'] / 10,  # Convert millibars to kPa
        'Rel Hum_%': current['humidity'],  # Relative humidity in %
        'Wind Speed_km/h': current['wind_kph'],  # Wind speed in km/h
        'Visibility_km': current['vis_km'],  # Visibility in km
        'Hour': datetime.strptime(current['last_updated'], '%Y-%m-%d %H:%M').hour  }
      
input_features = pd.DataFrame([weather_data])
input_values = input_features[['Temp_C', 'Press_kPa', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Hour']]

print(weather_data)

def test_predict(client):
    response = client.post('/predict', json={"features":input_values})
    assert response.status_code == 200
    assert "prediction" in response.get_json()



    