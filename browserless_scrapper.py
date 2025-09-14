import asyncio
import os
import io
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import json
import time
import urllib.parse
import random
# --- Config ---
MAX_IMAGES = 100
SCROLL_PAUSE = 2.0
DOWNLOAD_DELAY = 0.5
BATCH_SIZE = 5  # Process 5 foods before restarting browser
MAX_RETRIES = 3  # Retry failed items
PAGE_LOAD_TIMEOUT = 30

def load_food_data():
    try:
        with open('food_ranking_list_usa.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: food_ranking_list_usa.json not found!")
        return []
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON format in food_ranking_list_usa.json")
        return []


def ask_start_index(food_data):
    if not food_data:
        return 0
    while True:
        try:
            start_index = int(input(f"Enter start index (0 to {len(food_data)-1}): "))
            if 0 <= start_index < len(food_data):
                return start_index
            else:
                print(f"Invalid input. Please enter a number between 0 and {len(food_data)-1}")
        except ValueError:
            print("Please enter a valid integer.")

def download_image(download_path, url, file_name):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        resp = requests.get(url, timeout=15, headers=headers, stream=True)
        resp.raise_for_status()
        
        if 'image' not in resp.headers.get('content-type', '').lower():
            print(f"SKIPPED {file_name}: Not an image")
            return False
            
        image_file = io.BytesIO(resp.content)
        image = Image.open(image_file).convert("RGB")
        
        os.makedirs(download_path, exist_ok=True)
        file_path = os.path.join(download_path, file_name)
        
        image.save(file_path, "JPEG", quality=85)
        print(f"✓ Downloaded: {file_name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False
def create_driver():
    """Create a new Chrome WebDriver instance with optimized settings"""
    try:
        chrome_options = Options()
        
        # Performance optimizations
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-software-rasterizer')
        chrome_options.add_argument('--disable-background-timer-throttling')
        chrome_options.add_argument('--disable-backgrounding-occluded-windows')
        chrome_options.add_argument('--disable-renderer-backgrounding')
        chrome_options.add_argument('--disable-features=TranslateUI')
        chrome_options.add_argument('--disable-ipc-flooding-protection')
        
        # Memory optimizations
        chrome_options.add_argument('--memory-pressure-off')
        chrome_options.add_argument('--max_old_space_size=4096')
        
        # Disable unnecessary features
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--disable-images')  # We'll enable images specifically for our search
        chrome_options.add_argument('--disable-javascript')  # Most image loading doesn't need JS
        # User agent
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Window size
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Optional: Run headless (uncomment to hide browser window)
        # chrome_options.add_argument('--headless')
        
        # Create service (you may need to specify the path to chromedriver)
        # service = Service('/path/to/chromedriver')  # Uncomment and modify if needed
        service = Service()  # Uses chromedriver from PATH
        
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(10)
        
        return driver
        
    except Exception as e:
        print(f"Failed to create Chrome driver: {e}")
        print("Make sure you have Chrome and chromedriver installed.")
        print("You can install chromedriver using: pip install chromedriver-autoinstaller")
        return None

def scroll_down(driver):
    """Scroll down the page"""
    try:
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(SCROLL_PAUSE)
        return True
    except Exception as e:
        print(f"Error scrolling: {e}")
        return False

def get_image_urls_bing(driver, max_images):
    """Get image URLs from Bing with better error handling"""
    image_urls = set()
    scroll_attempts = 0
    max_scrolls = 8
   try:
        while len(image_urls) < max_images and scroll_attempts < max_scrolls:
            # Scroll to load more images
            if not scroll_down(driver):
                print("Failed to scroll, breaking")
                break
            scroll_attempts += 1
            
            # Get image elements
            try:
                # Wait for images to load
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'img.mimg'))
                )
                
                images = driver.find_elements(By.CSS_SELECTOR, 'img.mimg')
                
                for img in images:
                    if len(image_urls) >= max_images:
                        break
                        
                    try:
                        src = img.get_attribute('src')
                        if src and src.startswith('http') and src not in image_urls:
                            image_urls.add(src)
                            print(f"Found {len(image_urls)}/{max_images} images")
                            
                    except Exception:
                        continue
                        
            except TimeoutException:
                print("No more images found")
                break
            except Exception as e:
                print(f"Error getting images: {e}")
                break
                
    except Exception as e:
        print(f"Error in get_image_urls_bing: {e}")
    
    return list(image_urls)
def search_food_images_safe(driver, food_name, max_images):
    """Search for images with comprehensive error handling"""
    try:
        encoded_food = urllib.parse.quote(food_name)
        search_url = f"https://www.bing.com/images/search?q={encoded_food}+food+recipe"
        
        print(f"Navigating to: {search_url}")
        driver.get(search_url)
        
        # Wait for page to load
        time.sleep(3)
        
        # Wait for images to appear
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'img.mimg'))
            )
        except TimeoutException:
            print(f"No images found for {food_name} (timeout)")
            return []
        
        # Get image URLs
        image_urls = get_image_urls_bing(driver, max_images)
        return image_urls
        
    except Exception as e:
        print(f"Error searching for {food_name}: {e}")
        return []

def process_food_batch(food_data, start_idx, end_idx):
    """Process a batch of foods with a fresh browser connection"""
    driver = None
    batch_results = []
    
    try:
        print(f"\n=== Starting batch {start_idx}-{end_idx} ===")
        
        # Create browser driver
        driver = create_driver()
        if not driver:
            print("Failed to create browser driver for this batch")
            return batch_results
        
        # Process each food in the batch
        for idx in range(start_idx, min(end_idx, len(food_data))):
            food_name = food_data[idx]['name']
            print(f"\n[{idx+1}/{len(food_data)}] Processing: {food_name}")
            try:
                # Search for images
                image_urls = search_food_images_safe(driver, food_name, MAX_IMAGES)
                
                if not image_urls:
                    print(f"No images found for {food_name}")
                    batch_results.append({'food': food_name, 'success': False, 'count': 0})
                    continue
                
                print(f"Found {len(image_urls)} images for {food_name}")
                
                # Download images
                download_count = 0
                for i, url in enumerate(image_urls):
                    if download_image(f"imgs/{food_name}/", url, f"{i+1}.jpg"):
                        download_count += 1
                    
                    # Small delay between downloads
                    time.sleep(DOWNLOAD_DELAY)
                
                print(f"Downloaded {download_count}/{len(image_urls)} images for {food_name}")
                batch_results.append({'food': food_name, 'success': True, 'count': download_count})
                
                # Brief pause between foods
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {food_name}: {e}")
                batch_results.append({'food': food_name, 'success': False, 'count': 0})
    
    except Exception as e:
        print(f"Batch processing error: {e}")
    
    finally:
        # Clean up browser connection
        if driver:
            try:
                driver.quit()
                print(f"Closed browser for batch {start_idx}-{end_idx}")
            except:
                pass
    
    return batch_results

def main():
    # Load food data
    food_data = load_food_data()
    if not food_data:
        return

    start_index = ask_start_index(food_data)
    
    print(f"\nStarting from index {start_index}")
    print(f"Total foods to process: {len(food_data) - start_index}")
    print(f"Processing in batches of {BATCH_SIZE}")
    
    all_results = []
    
    # Process in batches
    for batch_start in range(start_index, len(food_data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(food_data))
        
        # Add random delay between batches to avoid rate limiting
        if batch_start > start_index:
            delay = random.randint(5, 15)
            print(f"\nWaiting {delay} seconds before next batch...")
            time.sleep(delay)
 # Process the batch
        batch_results = process_food_batch(food_data, batch_start, batch_end)
        all_results.extend(batch_results)
        
        # Print batch summary
        successful = sum(1 for r in batch_results if r['success'])
        total_downloads = sum(r['count'] for r in batch_results)
        print(f"\nBatch {batch_start}-{batch_end-1} completed:")
        print(f"  Successful: {successful}/{len(batch_results)}")
        print(f"  Total downloads: {total_downloads}")
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    successful_foods = sum(1 for r in all_results if r['success'])
    total_downloads = sum(r['count'] for r in all_results)
    
    print(f"Foods processed successfully: {successful_foods}/{len(all_results)}")
    print(f"Total images downloaded: {total_downloads}")
    
    # Show failed items
    failed_items = [r['food'] for r in all_results if not r['success']]
    if failed_items:
        print(f"\nFailed items ({len(failed_items)}):")
        for item in failed_items[:10]:  # Show first 10
            print(f"  - {item}")
        if len(failed_items) > 10:
            print(f"  ... and {len(failed_items) - 10} more")

if __name__ == "__main__":
    print("Food Image Scraper with Selenium")
    print("=" * 40)
    print("Features:")
    print("- Selenium WebDriver for better control")
    print("- Batch processing with browser restart")
    print("- Optimized Chrome settings")
    print("- Better error handling")
    print("- Progress tracking")
    print()
    
    # Check if chromedriver is available
    try:
        test_driver = create_driver()
        if test_driver:
            test_driver.quit()
            print("✓ Chrome WebDriver is available")
        else:
            print("✗ Chrome WebDriver setup failed")
            exit(1)
    except Exception as e:
        print(f"✗ Chrome WebDriver test failed: {e}")
        print("\nTo fix this:")
        print("1. Install Chrome browser")
        print("2. Install chromedriver: pip install chromedriver-autoinstaller")
        print("3. Or download chromedriver manually and add to PATH")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
