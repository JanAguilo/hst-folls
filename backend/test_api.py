"""
Test script for the backend API
Run this to verify the backend is working correctly
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Total events: {data['total_events']}")
            print(f"   Total commodities: {data['total_commodities_in_mapping']}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to backend: {e}")
        print("   Make sure the backend is running with: python backend/app.py")
        return False


def test_search_markets(commodity):
    """Test the search markets endpoint"""
    print(f"\n🔍 Testing search for commodity: {commodity}")
    try:
        response = requests.post(
            f"{BASE_URL}/api/search-markets",
            json={"commodity": commodity},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search successful!")
            print(f"   Message: {data['message']}")
            print(f"   Direct results: {len(data['directResults'])} markets")
            print(f"   Correlated commodity: {data['correlatedCommodity']}")
            print(f"   Correlated results: {len(data['correlatedResults'])} markets")
            
            # Print first market if available
            if data['directResults']:
                first_market = data['directResults'][0]
                print(f"\n   📊 Sample direct market:")
                print(f"      Question: {first_market['question']}")
                print(f"      YES: {first_market['yesPrice']}, NO: {first_market['noPrice']}")
                print(f"      Volume: ${first_market['volume']:,.2f}")
            
            if data['correlatedResults'] and not data['directResults']:
                first_market = data['correlatedResults'][0]
                print(f"\n   📊 Sample correlated market:")
                print(f"      Question: {first_market['question']}")
                print(f"      YES: {first_market['yesPrice']}, NO: {first_market['noPrice']}")
                print(f"      Volume: ${first_market['volume']:,.2f}")
            
            return True
        else:
            print(f"❌ Search failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("🧪 Backend API Test Suite")
    print("="*60)
    
    # Test health
    if not test_health():
        print("\n⚠️  Backend is not running or not accessible")
        print("   Start it with: python backend/app.py")
        exit(1)
    
    # Test commodity searches
    test_commodities = [
        "Gold (GC=F)",      # Should have direct results
        "gold",             # Alternative format
        "Wheat (ZW=F)",     # Should use correlated commodity (Oil)
        "Silver (SI=F)",    # Should have direct results
    ]
    
    all_passed = True
    for commodity in test_commodities:
        if not test_search_markets(commodity):
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)
