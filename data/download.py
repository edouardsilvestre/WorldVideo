import os
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    """
    Downloads a file from a URL to a local destination path with a progress bar.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        # Verify the file is not empty/corrupted
        if os.path.getsize(dest_path) >= 819200000:  # Full dataset is exactly 819,200,080 bytes
            print(f"File already exists and seems complete at {dest_path}. Skipping download.")
            return
        else:
            print(f"File at {dest_path} is incomplete or empty. Starting download/redownload.")

    print(f"Downloading Moving MNIST from {url} to {dest_path}...")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB chunks
        
        with open(dest_path, 'wb') as file, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="mnist_test_seq.npy"
        ) as bar:
            for data in response.iter_content(block_size):
                if data:
                    size = file.write(data)
                    bar.update(size)
        
        print(f"\nSuccessfully downloaded to {dest_path}!")
    except Exception as e:
        print(f"\nError during download: {e}")
        # Clean up partial file on failure
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e

if __name__ == "__main__":
    url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    
    # Establish project root and relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest_path = os.path.join(base_dir, "data", "raw", "mnist_test_seq.npy")
    
    download_file(url, dest_path)
