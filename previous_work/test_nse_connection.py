#!/usr/bin/env python3
"""
Test NSE Connection and Scraping
================================

This script tests the NSE connection and scraping functionality
to debug any issues with the web scraping.

Author: NSE Options Analysis Project
License: See LICENSE file
"""

import requests
import time
import random
import json

def test_nse_connection():
    """Test basic NSE connection"""
    print("Testing NSE Connection...")
    print("=" * 50)
    
    # Test 1: Basic home page access
    print("1. Testing basic home page access...")
    try:
        session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        resp = session.get("https://www.nseindia.com", headers=headers, timeout=15)
        print(f"   Status Code: {resp.status_code}")
        print(f"   Content Length: {len(resp.content)}")
        print(f"   Content Type: {resp.headers.get('content-type', 'Unknown')}")
        
        if resp.status_code == 200:
            print("   ✓ Successfully accessed NSE home page")
            
            # Check if we got the expected content
            if "National Stock Exchange" in resp.text or "NSE" in resp.text:
                print("   ✓ Content appears to be NSE website")
            else:
                print("   ⚠ Content doesn't look like NSE website")
                print(f"   First 200 chars: {resp.text[:200]}")
        else:
            print(f"   ✗ Failed to access NSE home page")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Test 2: Try to get cookies and session
    print("2. Testing session and cookies...")
    try:
        session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nseindia.com/option-chain",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }
        
        # First call to home page
        print("   Accessing home page to set cookies...")
        home_resp = session.get("https://www.nseindia.com", headers=headers, timeout=15)
        print(f"   Home page status: {home_resp.status_code}")
        
        # Check cookies
        cookies = session.cookies
        print(f"   Cookies received: {len(cookies)}")
        for cookie in cookies:
            print(f"     {cookie.name}: {cookie.value}")
        
        # Wait a bit
        delay = random.uniform(2, 4)
        print(f"   Waiting {delay:.1f}s...")
        time.sleep(delay)
        
        # Try the API endpoint
        print("   Trying API endpoint...")
        api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        api_resp = session.get(api_url, headers=headers, timeout=15)
        
        print(f"   API Status Code: {api_resp.status_code}")
        print(f"   API Response Headers: {dict(api_resp.headers)}")
        
        if api_resp.status_code == 200:
            print("   ✓ Successfully accessed NSE API")
            try:
                data = api_resp.json()
                print(f"   ✓ Got JSON data with {len(str(data))} characters")
                
                # Check if it looks like options data
                if "records" in data and "data" in data.get("records", {}):
                    print("   ✓ Data structure looks like options data")
                    records = data["records"]
                    if "underlyingValue" in records:
                        print(f"   ✓ Spot price: {records['underlyingValue']}")
                    if "expiryDates" in records:
                        print(f"   ✓ Expiry dates: {len(records['expiryDates'])} available")
                        for i, exp in enumerate(records['expiryDates'][:5]):  # Show first 5
                            print(f"     {i+1}. {exp}")
                else:
                    print("   ⚠ Data structure doesn't look like expected options data")
                    print(f"   Keys in data: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
            except json.JSONDecodeError as e:
                print(f"   ✗ Failed to parse JSON: {e}")
                print(f"   Response text: {api_resp.text[:500]}")
        else:
            print(f"   ✗ Failed to access NSE API")
            print(f"   Response text: {api_resp.text[:500]}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 3: Try alternative endpoints
    print("3. Testing alternative endpoints...")
    alternative_endpoints = [
        "https://www.nseindia.com/api/option-chain-indices",
        "https://www.nseindia.com/api/option-chain-equities",
        "https://www.nseindia.com/api/option-chain",
    ]
    
    for endpoint in alternative_endpoints:
        try:
            print(f"   Trying: {endpoint}")
            resp = session.get(endpoint, headers=headers, timeout=15)
            print(f"     Status: {resp.status_code}")
            if resp.status_code == 200:
                print(f"     ✓ Success! Content length: {len(resp.content)}")
                try:
                    data = resp.json()
                    print(f"     ✓ JSON parsed successfully")
                except:
                    print(f"     ⚠ Not valid JSON")
            else:
                print(f"     ✗ Failed")
        except Exception as e:
            print(f"     ✗ Error: {e}")
    
    print()
    
    # Test 4: Try with different headers
    print("4. Testing with different headers...")
    different_headers = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }
    ]
    
    for i, test_headers in enumerate(different_headers):
        try:
            print(f"   Testing header set {i+1}...")
            test_session = requests.Session()
            
            # First get home page
            home_resp = test_session.get("https://www.nseindia.com", headers=test_headers, timeout=15)
            print(f"     Home page status: {home_resp.status_code}")
            
            if home_resp.status_code == 200:
                # Wait a bit
                time.sleep(random.uniform(1, 2))
                
                # Try API
                api_resp = test_session.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", 
                                          headers=test_headers, timeout=15)
                print(f"     API status: {api_resp.status_code}")
                
                if api_resp.status_code == 200:
                    print(f"     ✓ Success with header set {i+1}!")
                    break
            else:
                print(f"     ✗ Home page failed")
                
        except Exception as e:
            print(f"     ✗ Error: {e}")

def main():
    """Run the connection tests"""
    print("NSE CONNECTION TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_nse_connection()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 