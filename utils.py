import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import re

driver = webdriver.Chrome()

if not os.path.exists('face_images'):
    os.makedirs('face_images')

def sanitize_filename(url):
    base_name = url.split('/')[-1].split('?')[0]
    base_name = re.sub(r'[^\w\-\.]', '_', base_name)
    if not base_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        base_name += '.jpg'
    return base_name[:100]

def download_image(url, folder_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            filename = sanitize_filename(url)
            with open(os.path.join(folder_path, filename), 'wb') as file:
                file.write(response.content)
            print(f"Image {filename} saved!")
        else:
            print(f"Failed to retrieve image from {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

driver.get("https://unsplash.com/s/photos/people")

time.sleep(3)

for _ in range(5):
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(2)

img_elements = driver.find_elements(By.TAG_NAME, 'img')

count = 0
for img in img_elements:
    img_url = img.get_attribute('src')
    if img_url and 'https' in img_url and 'unsplash.com' in img_url:  
        download_image(img_url, 'face_images')
        count += 1
    if count >= 100:
        break

driver.quit()

print(f"Downloaded {count} images.")