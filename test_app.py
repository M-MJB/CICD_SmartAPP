import pytest
from app import app


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_predict(client):
    response = client.post('/predict', json={"features":[0.00632,18,2.31,0.0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98]})
    assert response.status_code == 200
    assert "prediction" in response.get_json()