"""
VisionaryAI API Security Testing Script
Step 4: Test the deployed service with security features
"""

import requests
import time
from concurrent.futures import ThreadPoolExecutor

# API configuration
BASE_URL = "http://localhost:8000"
VALID_API_KEY = "visionaryai-demo-key"
INVALID_API_KEY = "invalid-key-123"

# Headers for authenticated requests
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_API_KEY}"}
INVALID_AUTH_HEADERS = {"Authorization": f"Bearer {INVALID_API_KEY}"}


def test_authentication():
    """Test API key authentication"""
    print("🔐 Testing Authentication...")

    # Test without API key
    print("\n--- Test 1: No API Key ---")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code} (Expected: 401/403)")

    # Test with invalid API key
    print("\n--- Test 2: Invalid API Key ---")
    response = requests.get(f"{BASE_URL}/", headers=INVALID_AUTH_HEADERS)
    print(f"Status: {response.status_code} (Expected: 401)")
    if response.status_code == 401:
        print("✅ Invalid API key correctly rejected")

    # Test with valid API key
    print("\n--- Test 3: Valid API Key ---")
    response = requests.get(f"{BASE_URL}/", headers=AUTH_HEADERS)
    print(f"Status: {response.status_code} (Expected: 200)")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Authenticated as: {result.get('user', 'Unknown')}")

    return response.status_code == 200


def test_input_validation():
    """Test input sanitization and validation"""
    print("\n🛡️ Testing Input Validation...")

    test_cases = [
        {
            "name": "Empty text",
            "data": {"text": "", "customer_id": "test_001"},
            "expected_status": 422,
        },
        {
            "name": "Too long text",
            "data": {"text": "x" * 1001, "customer_id": "test_002"},
            "expected_status": 422,
        },
        {
            "name": "HTML injection attempt",
            "data": {
                "text": "Help me <script>alert('xss')</script>",
                "customer_id": "test_003",
            },
            "expected_status": 200,  # Should be sanitized and processed
        },
        {
            "name": "Invalid customer ID format",
            "data": {"text": "Valid ticket text", "customer_id": "invalid@customer.id"},
            "expected_status": 422,
        },
        {
            "name": "Invalid priority",
            "data": {"text": "Valid ticket text", "priority": "super_urgent"},
            "expected_status": 422,
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        response = requests.post(
            f"{BASE_URL}/classify", json=test_case["data"], headers=AUTH_HEADERS
        )
        print(
            f"Status: {response.status_code} (Expected: {test_case['expected_status']})"
        )

        if response.status_code == test_case["expected_status"]:
            print("✅ Input validation working correctly")
        else:
            print(f"❌ Unexpected response: {response.text[:200]}")


def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n⏱️ Testing Rate Limiting...")

    def make_request(request_id):
        """Make a single request"""
        try:
            response = requests.get(f"{BASE_URL}/", headers=AUTH_HEADERS, timeout=5)
            return {"id": request_id, "status": response.status_code}
        except Exception as e:
            return {"id": request_id, "error": str(e)}

    # Make rapid requests to trigger rate limiting
    print("Making 15 rapid requests (limit: 10/minute)...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(15)]
        results = [future.result() for future in futures]

    # Count responses
    success_count = len([r for r in results if r.get("status") == 200])
    rate_limited = len([r for r in results if r.get("status") == 429])

    print(f"Successful requests: {success_count}")
    print(f"Rate limited (429): {rate_limited}")

    if rate_limited > 0:
        print("✅ Rate limiting is working")
    else:
        print("⚠️ Rate limiting may not be configured correctly")


def test_authorization_tiers():
    """Test different user tier access levels"""
    print("\n👥 Testing Authorization Tiers...")

    # Test model info access with basic tier
    print("\n--- Basic Tier Model Info ---")
    response = requests.get(f"{BASE_URL}/model/info", headers=AUTH_HEADERS)
    if response.status_code == 200:
        info = response.json()
        # Basic tier should get limited info
        if "model_metadata" not in info or len(info) <= 3:
            print("✅ Basic tier receives limited model info")
        else:
            print("⚠️ Basic tier might be getting too much info")

    # Test batch processing (should be limited for basic tier)
    print("\n--- Basic Tier Batch Access ---")
    batch_data = [{"text": "Test ticket 1"}, {"text": "Test ticket 2"}]
    response = requests.post(
        f"{BASE_URL}/classify/batch", json=batch_data, headers=AUTH_HEADERS
    )

    if response.status_code == 403:
        print("✅ Basic tier correctly denied batch access")
    elif response.status_code == 200:
        print("⚠️ Basic tier has batch access (check if intended)")
    else:
        print(f"Unexpected status: {response.status_code}")


def test_security_monitoring():
    """Test security monitoring endpoint"""
    print("\n📊 Testing Security Monitoring...")

    response = requests.get(f"{BASE_URL}/security/stats", headers=AUTH_HEADERS)

    if response.status_code == 403:
        print("✅ Security stats correctly restricted to internal users")
    elif response.status_code == 200:
        print("⚠️ Security stats accessible to basic tier (potential security issue)")
    else:
        print(f"Unexpected status: {response.status_code}")


def test_error_handling():
    """Test that errors don't leak sensitive information"""
    print("\n🚨 Testing Secure Error Handling...")

    # Test with malformed JSON
    print("\n--- Malformed JSON ---")
    response = requests.post(
        f"{BASE_URL}/classify", data="invalid json", headers=AUTH_HEADERS
    )

    if response.status_code == 422:  # Unprocessable Entity
        error_detail = response.json().get("detail", "")
        # Should not expose internal paths or sensitive info
        if (
            "models/" not in str(error_detail)
            and "traceback" not in str(error_detail).lower()
        ):
            print("✅ Error messages don't leak sensitive information")
        else:
            print("⚠️ Error messages might be exposing too much information")


def test_logging_privacy():
    """Verify that sensitive data isn't logged in plain text"""
    print("\n📝 Testing Privacy in Logging...")

    # Make a request with sensitive-looking data
    sensitive_data = {
        "text": "My credit card number is 1234-5678-9012-3456 and my SSN is 123-45-6789",
        "customer_id": "sensitive_customer_123",
    }

    response = requests.post(
        f"{BASE_URL}/classify", json=sensitive_data, headers=AUTH_HEADERS
    )

    if response.status_code == 200:
        result = response.json()
        # Check that sensitive data isn't echoed back in response
        response_text = str(result)
        if "1234-5678" not in response_text and "123-45-6789" not in response_text:
            print("✅ Sensitive data not echoed in response")
        else:
            print("⚠️ Sensitive data found in response")

    print(
        "Note: Check server logs manually to ensure sensitive data isn't logged in plain text"
    )


def main():
    """Run all security tests"""
    print("🚀 Starting VisionaryAI Security Tests...")
    print("=" * 60)

    try:
        # Wait for service
        print("⏳ Waiting for service to start...")
        time.sleep(3)

        # Run security tests
        auth_ok = test_authentication()
        if not auth_ok:
            print("❌ Authentication failed, skipping further tests")
            return

        test_input_validation()
        test_rate_limiting()
        test_authorization_tiers()
        test_security_monitoring()
        test_error_handling()
        test_logging_privacy()

        print("\n" + "=" * 60)
        print("🎉 Security testing completed!")
        print(
            "📋 Review any warnings above and check server logs for privacy compliance"
        )

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the service running?")
    except Exception as e:
        print(f"❌ Test error: {str(e)}")


if __name__ == "__main__":
    main()
