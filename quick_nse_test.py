#!/usr/bin/env python3
"""
Quick NSE Connection Test
=========================

Fast test to check NSE connection status without long delays.

Author: NSE Options Analysis Project
License: See LICENSE file
"""

import requests

def quick_test():
    """Quick test of NSE connection"""
    print("Quick NSE Connection Test")
    print("=" * 40)
    
    try:
        # Test 1: Basic home page (fast)
        print("1. Testing NSE home page...")
        resp = requests.get("https://www.nseindia.com", timeout=5)
        print(f"   Status: {resp.status_code}")
        print(f"   Content length: {len(resp.content)}")
        
        if resp.status_code == 200:
            print("   ✓ Home page accessible")
            if "NSE" in resp.text[:500]:
                print("   ✓ Content looks like NSE")
            else:
                print("   ⚠ Content doesn't look like NSE")
        else:
            print("   ✗ Home page failed")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    try:
        # Test 2: Try API endpoint (fast)
        print("2. Testing NSE API endpoint...")
        api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        api_resp = requests.get(api_url, timeout=5)
        print(f"   Status: {api_resp.status_code}")
        
        if api_resp.status_code == 200:
            print("   ✓ API accessible")
            try:
                data = api_resp.json()
                print(f"   ✓ JSON data received")
                if "records" in data:
                    print("   ✓ Looks like options data")
                else:
                    print("   ⚠ Doesn't look like options data")
            except:
                print("   ⚠ Not valid JSON")
        else:
            print(f"   ✗ API failed with status {api_resp.status_code}")
            print(f"   Response: {api_resp.text[:200]}")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")

if __name__ == "__main__":
    quick_test() 