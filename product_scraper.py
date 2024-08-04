#!/usr/bin/env python
# coding: utf-8


import requests
import os
import csv

# This script fetches product data from Zara's API for different categories,
# saves the product information to CSV files, and downloads product images.

# API keys for authentication
API_KEYS = {
    "women": "your_api_key_for_women",
    "men": "your_api_key_for_men",
    "girls": "your_api_key_for_girls",
    "boys": "your_api_key_for_boys"
}

# URLs for fetching category data
CATEGORY_URLS = {
    "women": "https://www.zara.com/ua/en/category/2419517/products?ajax=tru",
    "men": "https://www.zara.com/ua/en/category/2443335/products?ajax=tru",
    "girls": "https://www.zara.com/ua/en/category/2426193/products?ajax=tru",
    "boys": "https://www.zara.com/ua/en/category/2426702/products?ajax=tru"
}

# Headers to mimic a browser request and include API keys
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Authorization": "Bearer " + API_KEYS["women"]  # Example for the 'women' category
}

# Files to save product information
CSV_FILES = {
    "women": 'women_products_info.csv',
    "men": 'men_products_info.csv',
    "girls": 'girls_products_info.csv',
    "boys": 'boys_products_info.csv'
}

# Directories to save images
IMAGE_DIRS = {
    "women": 'zara_images/women',
    "men": 'zara_images/men',
    "girls": 'zara_images/girls',
    "boys": 'zara_images/boys'
}

for category in IMAGE_DIRS:
    os.makedirs(IMAGE_DIRS[category], exist_ok=True)

def fetch_data_from_api(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def save_image(image_url, image_dir, product_name):
    try:
        image_response = requests.get(image_url)
        image_response.raise_for_status()  # Check for HTTP errors

        # Get the file name from the URL
        image_name = os.path.join(image_dir, image_url.split('/')[-1].split('?')[0])
        
        # Save the image to disk
        with open(image_name, 'wb') as img_file:
            img_file.write(image_response.content)
        
        print(f"Image for product '{product_name}' saved as {image_name}")
        return image_name
    except Exception as img_err:
        print(f"Error downloading image for product '{product_name}': {img_err}")
        return None

def save_product_info_to_csv(products, csv_file, image_dir):
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Brand', 'Product Name', 'Category', 'Product ID', 'Color', 'Style', 'Price', 'Image URL', 'Gender', 'Product URL']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for product in products:
            brand = product.get('brand', {}).get('brandGroupCode', 'No brand')
            name = product.get('name', 'No name')
            category = product.get('familyName', 'No category')
            product_id = product.get('reference', 'No ID')
            gender = product.get('sectionName', 'No gender')
            
            # Determine the price
            price_info = product.get('price', {})
            price = price_info.get('formatted', 'No price') if isinstance(price_info, dict) else price_info
            
            # Extract the image URL
            image_url = None
            colors = product.get('detail', {}).get('colors', [])
            if colors:
                xmedia = colors[0].get('xmedia', [])
                for media in xmedia:
                    if media.get('type') == 'image':
                        image_url = media.get('extraInfo', {}).get('deliveryUrl')
                        break
            
            color = ', '.join([color.get('name', 'No color') for color in colors])
            style = colors[0].get('outfitId', 'No style') if colors else 'No style'
            
            seo = product.get('seo', {})
            seo_keyword = seo.get('keyword', 'no-keyword')
            seo_product_id = seo.get('seoProductId', 'no-id')
            seo_discern_product_id = seo.get('discernProductId', 'no-id')
            
            product_url = f"https://www.zara.com/ua/en/{seo_keyword}-p{seo_product_id}.html?v1={seo_discern_product_id}"
            
            writer.writerow({
                'Brand': brand,
                'Product Name': name,
                'Category': category,
                'Product ID': product_id,
                'Color': color,
                'Style': style,
                'Price': price,
                'Image URL': image_url,
                'Gender': gender,
                'Product URL': product_url
            })
            
            if image_url:
                save_image(image_url, image_dir, name)
            else:
                print(f"Image for product '{name}' not found")

def main():
    for category, url in CATEGORY_URLS.items():
        # Update the Authorization header for each category
        HEADERS["Authorization"] = "Bearer " + API_KEYS[category]
        
        # Fetch data from the API
        data = fetch_data_from_api(url, HEADERS)
        
        if data:
            # Check for products in the response
            product_groups = data.get('productGroups', [])
            products = []
            
            for group in product_groups:
                elements = group.get('elements', [])
                for element in elements:
                    commercial_components = element.get('commercialComponents', [])
                    products.extend(commercial_components)
            
            # Save product information to CSV
            save_product_info_to_csv(products, CSV_FILES[category], IMAGE_DIRS[category])

if __name__ == "__main__":
    main()






