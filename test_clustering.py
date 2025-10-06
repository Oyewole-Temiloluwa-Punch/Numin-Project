"""
Test script for the Stock Market Pattern Clustering API
"""

import requests
import json
from datetime import date, datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_adaptive_clustering():
    """Test the adaptive clustering algorithm with custom points"""
    print("\nTesting adaptive clustering algorithm...")
    try:
        # Test with sample points
        test_points = "1.0,1.2,1.1,2.5,2.6,2.4,5.0"
        response = requests.get(
            f"{BASE_URL}/clustering/adaptive-cluster",
            params={
                "points": test_points,
                "base_gap": 0.3,
                "max_gap": 1.5
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Input points: {data.get('input_points')}")
            print(f"Cluster mean: {data.get('cluster_mean')}")
        else:
            print(f"Error response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_clustering_statistics():
    """Test clustering statistics endpoint"""
    print("\nTesting clustering statistics...")
    try:
        response = requests.get(f"{BASE_URL}/clustering/statistics")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            stats = data.get('data_statistics', {})
            print(f"Total records: {stats.get('total_records')}")
            print(f"Date range: {stats.get('date_range', {}).get('start')} to {stats.get('date_range', {}).get('end')}")
        else:
            print(f"Error response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_clustered_projections():
    """Test clustered projections endpoint"""
    print("\nTesting clustered projections...")
    try:
        response = requests.get(
            f"{BASE_URL}/clustering/projections",
            params={
                "category": "All",
                "bar_count_filter": "All",
                "recent_bars": 3,  # Smaller for testing
                "max_overlay": 2,
                "look_ahead": 10
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            
            print(f"Pattern dates: {len(data.get('pattern_dates', []))}")
            projections = data.get('projections', {})
            print(f"Total projection days: {len(projections)}")
            
            # Show sectioned data
            sectioned_data = data.get('sectioned_data', {})
            print(f"Sectioned data windows: {len(sectioned_data)}")
            
            # Show first few projections as examples
            if projections:
                first_day = list(projections.keys())[0]
                first_day_data = projections[first_day]
                print(f"First day ({first_day}): Positive={first_day_data.get('positive')}, Negative={first_day_data.get('negative')}, Cluster={first_day_data.get('cluster')}")
            
            # Show sectioned data examples
            if sectioned_data:
                window_5 = sectioned_data.get('window_5', {})
                window_10 = sectioned_data.get('window_10', {})
                print(f"Window 5: Highest Positive={window_5.get('highest_positive')}, Lowest Negative={window_5.get('lowest_negative')}, Cluster Change={window_5.get('cluster_change_from_base')}")
                print(f"Window 10: Highest Positive={window_10.get('highest_positive')}, Lowest Negative={window_10.get('lowest_negative')}, Cluster Change={window_10.get('cluster_change_from_base')}")
        else:
            print(f"Error response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_clustered_projections_with_categories():
    """Test clustered projections with different categories"""
    print("\nTesting clustered projections with categories...")
    try:
        categories = ["All", "7D+", "7D-", "Tech+", "Tech-"]
        results = {}
        
        for category in categories:
            response = requests.get(
                f"{BASE_URL}/clustering/projections",
                params={
                    "category": category,
                    "bar_count_filter": "All",
                    "recent_bars": 3,
                    "max_overlay": 2,
                    "look_ahead": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                projections = data.get('projections', {})
                sectioned_data = data.get('sectioned_data', {})
                results[category] = {
                    "projection_days": len(projections),
                    "pattern_dates": len(data.get('pattern_dates', [])),
                    "sectioned_windows": len(sectioned_data)
                }
                print(f"Category {category}: {results[category]}")
            else:
                print(f"Error for category {category}: {response.status_code}")
                results[category] = None
        
        return all(results.values())
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_clustering_with_different_parameters():
    """Test clustering with different gap parameters"""
    print("\nTesting clustering with different parameters...")
    try:
        test_points = "1.0,1.1,1.2,1.3,5.0,5.1,5.2,10.0"
        
        # Test with tight clustering
        response1 = requests.get(
            f"{BASE_URL}/clustering/adaptive-cluster",
            params={
                "points": test_points,
                "base_gap": 0.1,
                "max_gap": 0.5
            }
        )
        
        # Test with loose clustering
        response2 = requests.get(
            f"{BASE_URL}/clustering/adaptive-cluster",
            params={
                "points": test_points,
                "base_gap": 1.0,
                "max_gap": 3.0
            }
        )
        
        print(f"Tight clustering status: {response1.status_code}")
        print(f"Loose clustering status: {response2.status_code}")
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            print(f"Tight clustering mean: {data1.get('cluster_mean')}")
            print(f"Loose clustering mean: {data2.get('cluster_mean')}")
            
            return True
        else:
            print(f"Error in parameter testing")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Stock Market Pattern Clustering API Tests")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Adaptive Clustering", test_adaptive_clustering),
        ("Clustering Statistics", test_clustering_statistics),
        ("Clustered Projections", test_clustered_projections),
        ("Category Projections", test_clustered_projections_with_categories),
        ("Different Parameters", test_clustering_with_different_parameters),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check the API server and data files.")

if __name__ == "__main__":
    main()
