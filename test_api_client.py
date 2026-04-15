"""
Test client for Wan2GP API.
Tests job submission, status tracking, and download.
"""
import requests
import time
import sys
import os

_port = os.getenv("WAN2GP_PORT", "8000")
API_URL = os.getenv("WAN2GP_URL", f"http://69.197.145.4:{_port}")
API_KEY = os.getenv("WAN2GP_API_KEY", "mypassword1234")
HEADERS = {"X-API-Key": API_KEY}

def test_health():
    """Test health endpoint."""
    print("=== Testing Health ===")
    response = requests.get(f"{API_URL}/health", headers=HEADERS)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_list_loras():
    """Test LoRA listing."""
    print("\n=== Testing LoRA List ===")
    response = requests.get(f"{API_URL}/loras", headers=HEADERS)
    print(f"LoRAs: {response.json()}")
    return response.status_code == 200

def test_submit_job():
    """Test job submission."""
    print("\n=== Testing Job Submission ===")
    job_request = {
        "prompt": "A smooth cinematic drone shot of a futuristic city with neon lights and flying vehicles, photorealistic, 4K quality",
        "resolution": "1280x720",
        "video_length": 81,
        "seed": -1,
        "steps": 8,
        "loras": {"ltx-2-19b-ic-lora-detailer.safetensors": 0.8}
    }
    
    response = requests.post(f"{API_URL}/generate", json=job_request, headers=HEADERS)
    print(f"Status: {response.status_code}")
    job_data = response.json()
    print(f"Job ID: {job_data['job_id']}")
    print(f"Queue Position: {job_data['queue_position']}")
    print(f"Estimated Wait: {job_data['estimated_wait_minutes']} min")
    
    return job_data['job_id']

def test_poll_status(job_id, timeout_minutes=30):
    """Poll job status until completion."""
    print(f"\n=== Polling Status for {job_id} ===")
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"Timeout after {timeout_minutes} minutes")
            return None
        
        response = requests.get(f"{API_URL}/status/{job_id}", headers=HEADERS)
        status_data = response.json()
        
        status = status_data['status']
        print(f"[{int(elapsed)}s] Status: {status}")
        
        if status == 'completed':
            print(f"Duration: {status_data.get('duration_seconds')} seconds")
            print(f"Output File: {status_data.get('output_file')}")
            return status_data
        
        elif status in ['failed', 'error']:
            print(f"Error: {status_data.get('error')}")
            return None
        
        time.sleep(10)

def test_download(job_id):
    """Test video download."""
    print(f"\n=== Downloading Video for {job_id} ===")
    response = requests.get(f"{API_URL}/download/{job_id}", headers=HEADERS)
    
    if response.status_code == 200:
        filename = f"{job_id}.mp4"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename} ({len(response.content)} bytes)")
        return True
    else:
        print(f"Download failed: {response.status_code}")
        print(response.json())
        return False

def test_queue_status():
    """Test queue status."""
    print("\n=== Testing Queue Status ===")
    response = requests.get(f"{API_URL}/queue", headers=HEADERS)
    queue_data = response.json()
    print(f"Total Jobs: {queue_data['total_jobs']}")
    print(f"Queue Length: {queue_data['queue_length']}")
    print(f"Running: {queue_data['running']}")
    print(f"Completed: {queue_data['completed']}")
    print(f"Errors: {queue_data['errors']}")

def test_list_jobs():
    """Test job listing."""
    print("\n=== Testing Job List ===")
    response = requests.get(f"{API_URL}/jobs/list?limit=10", headers=HEADERS)
    jobs_data = response.json()
    print(f"Recent jobs: {len(jobs_data['jobs'])}")
    for job in jobs_data['jobs'][:5]:
        print(f"  {job['job_id']}: {job['status']} - {job['prompt'][:50]}...")

def main():
    print("Wan2GP API Test Suite")
    print("=" * 50)
    
    # Test 1: Health
    if not test_health():
        print("Health check failed. Is the API running?")
        sys.exit(1)
    
    # Test 2: List LoRAs
    test_list_loras()
    
    # Test 3: Queue status
    test_queue_status()
    
    # Test 4: Submit job
    job_id = test_submit_job()
    
    # Test 5: Poll status
    status_data = test_poll_status(job_id, timeout_minutes=30)
    
    if status_data:
        # Test 6: Download
        test_download(job_id)
    
    # Test 7: Final queue check
    test_queue_status()
    
    # Test 8: List jobs
    test_list_jobs()
    
    print("\n=== Test Suite Complete ===")

if __name__ == "__main__":
    main()
