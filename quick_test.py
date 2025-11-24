"""Quick test to verify the API works"""

import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from fastapi.testclient import TestClient
from app.main import app, load_models

# Manually trigger model loading (TestClient doesn't auto-trigger startup in some versions)
print("Loading models for testing...")
load_models()
print()

client = TestClient(app)

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    print(f"Health check: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_models():
    """Test models endpoint"""
    response = client.get("/models")
    print(f"\nModels endpoint: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_generate():
    """Test generate endpoint"""
    response = client.post(
        "/generate",
        json={"model_type": "GAN", "num_samples": 1}
    )
    print(f"\nGenerate endpoint: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Generated {data['num_samples']} image(s)")
        print(f"Image data length: {len(data['images'][0])} characters")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def main():
    print("=" * 60)
    print("Quick API Test (using TestClient)")
    print("=" * 60)
    
    results = []
    results.append(("Health", test_health()))
    results.append(("Models", test_models()))
    results.append(("Generate", test_generate()))
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())

