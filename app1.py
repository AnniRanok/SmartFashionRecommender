#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# app.py
from flask import Flask, request, render_template, send_file
import numpy as np
import pandas as pd
import cv2
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
import os
import json

app = Flask(__name__)

# Load precomputed features and filenames
with open('featurevector_resnet.pkl', 'rb') as f:
    feature_list = np.array(pickle.load(f))

with open('filenames_resnet.pkl', 'rb') as f:
    filename = pickle.load(f)

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Load product metadata
df = pd.read_csv('info.csv')

def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_image = preprocess_input(expand_img)
    result = model.predict(pre_image).flatten()
    normalized = result / np.linalg.norm(result)
    return normalized

def recommend(features, feature_list):
    feature_list_np = np.array(feature_list)
    similarities = cosine_similarity([features], feature_list_np)
    top_indices = np.argsort(similarities[0])[-5:][::-1]
    return top_indices, similarities[0]

def recommend_similar_images(selected_image_path, model, feature_list, filename):
    selected_features = extract_feature(selected_image_path, model)
    top_indices, similarities = recommend(selected_features, feature_list)
    return top_indices, similarities

def save_similar_images_as_json(selected_image_path, top_indices, filename, similarities, df):
    """
    Save the metadata of the most similar images to a JSON file.
    """
    similar_images_info = []

    for i, idx in enumerate(top_indices):
        img_path = filename[idx]
        
        # Extract image information from the DataFrame
        image_name = df.loc[df['Image Name'] == img_path.split('/')[-1], 'Product Name'].values
        brand = df.loc[df['Image Name'] == img_path.split('/')[-1], 'Brand'].values
        product_url = df.loc[df['Image Name'] == img_path.split('/')[-1], 'Product URL'].values
        
        # Construct the metadata dictionary
        metadata = {
            "Product URL": product_url[0] if len(product_url) > 0 else "",
            "Product Name": image_name[0] if len(image_name) > 0 else "",
            "Brand": brand[0] if len(brand) > 0 else "",
            "Similarity": float(similarities[idx])
        }
        
        similar_images_info.append(metadata)

    # Save to JSON file
    output_file = 'static/similar_images_info.json'
    with open(output_file, 'w') as f:
        json.dump(similar_images_info, f, indent=4)
    
    return output_file


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            
            # Extract features and recommend similar images
            top_indices, similarities = recommend_similar_images(filepath, model, feature_list, filename)
            
            # Save metadata to JSON file
            json_file = save_similar_images_as_json(filepath, top_indices, filename, similarities, df)
            
            # Send the JSON file as a response
            return send_file(json_file, mimetype='application/json')
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

