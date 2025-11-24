"""Test script for the generative models API"""

import requests
import base64
from PIL import Image
import io
import time


def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_list_models():
    """Test list models endpoint"""
    print("\nTesting list models...")
    response = requests.get("http://localhost:8000/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_generate_gan():
    """Test GAN generation"""
    print("\nTesting GAN generation...")
    response = requests.post(
        "http://localhost:8000/generate",
        json={"model_type": "GAN", "num_samples": 2}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Generated {data['num_samples']} images")
        
        # Save first image
        img_data = base64.b64decode(data['images'][0])
        img = Image.open(io.BytesIO(img_data))
        img.save("test_gan.png")
        print("Saved test image to test_gan.png")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_generate_ebm():
    """Test EBM generation"""
    print("\nTesting EBM generation...")
    response = requests.post(
        "http://localhost:8000/generate",
        json={"model_type": "EBM", "num_samples": 2}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Generated {data['num_samples']} images")
        
        # Save first image
        img_data = base64.b64decode(data['images'][0])
        img = Image.open(io.BytesIO(img_data))
        img.save("test_ebm.png")
        print("Saved test image to test_ebm.png")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_generate_diffusion():
    """Test Diffusion generation"""
    print("\nTesting Diffusion generation...")
    response = requests.post(
        "http://localhost:8000/generate",
        json={"model_type": "Diffusion", "num_samples": 2}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Generated {data['num_samples']} images")
        
        # Save first image
        img_data = base64.b64decode(data['images'][0])
        img = Image.open(io.BytesIO(img_data))
        img.save("test_diffusion.png")
        print("Saved test image to test_diffusion.png")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_get_sample():
    """Test sample endpoint"""
    print("\nTesting sample endpoint...")
    response = requests.get("http://localhost:8000/sample/GAN")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        img.save("test_sample.png")
        print("Saved sample image to test_sample.png")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Generative Models API")
    print("=" * 60)
    
    # Wait for server to start
    print("\nWaiting for server to start...")
    for i in range(10):
        try:
            requests.get("http://localhost:8000/health", timeout=1)
            print("Server is ready!")
            break
        except:
            time.sleep(1)
            if i == 9:
                print("Error: Server did not start in time")
                return
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("List Models", test_list_models()))
    
    # Only test models that are available
    response = requests.get("http://localhost:8000/models")
    available_models = response.json().get("available_models", [])
    
    if "GAN" in available_models:
        results.append(("Generate GAN", test_generate_gan()))
    
    if "EBM" in available_models:
        results.append(("Generate EBM", test_generate_ebm()))
    
    if "Diffusion" in available_models:
        results.append(("Generate Diffusion", test_generate_diffusion()))
    
    if available_models:
        results.append(("Get Sample", test_get_sample()))
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()

