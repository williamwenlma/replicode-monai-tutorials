import requests

def test_basic_endpoints():
    base_url = "http://localhost:8015"
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure it's running.")
        return

    # Test model status endpoint
    print("\nTesting model status endpoint...")
    try:
        response = requests.get(f"{base_url}/model-status")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_basic_endpoints()