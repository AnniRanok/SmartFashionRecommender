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
import matplotlib.pyplot as plt

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

def save_similar_images(selected_image_path, top_indices, filename, similarities, df):
    """
    Save the selected image and the most similar images with metadata to a file.
    """
    # Create a subplot for the selected image and the similar images
    fig, axes = plt.subplots(1, len(top_indices) + 1, figsize=(20, 5))
    
    # Display the selected image
    selected_img = cv2.imread(selected_image_path)
    selected_img = cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(selected_img)
    axes[0].set_title("Selected Image")
    axes[0].axis('off')
    
    # Display similar images
    for i, idx in enumerate(top_indices):
        img_path = filename[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image
        axes[i + 1].imshow(img)
        axes[i + 1].axis('off')
        
        # Display the similarity score as the title
        axes[i + 1].set_title(f"Similarity: {similarities[idx]:.2f}", fontsize=12)
        
        # Extract image information from the DataFrame
        image_name = df.loc[df['Image Name'] == img_path.split('/')[-1], 'Product Name'].values
        brand = df.loc[df['Image Name'] == img_path.split('/')[-1], 'Brand'].values
        product_url = df.loc[df['Image Name'] == img_path.split('/')[-1], 'Product URL'].values
        
        # Construct the metadata text
        metadata_text = ""
        if len(image_name) > 0:
            metadata_text += f"Name: {image_name[0]}\n"
        if len(brand) > 0:
            metadata_text += f"Brand: {brand[0]}\n"
        
        # Add metadata text below the image with smaller font size
        axes[i + 1].text(0.5, -0.15, metadata_text, fontsize=8, ha='center', va='top', transform=axes[i + 1].transAxes)
        
        # Add clickable link text below the metadata
        if len(product_url) > 0:
            # Create a shorter display text for the URL
            link_text = "View on Website"
            # Add clickable-like text below the metadata
            axes[i + 1].text(0.5, -0.30, link_text, fontsize=10, ha='center', va='top', transform=axes[i + 1].transAxes, color='blue', style='italic', bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

    plt.tight_layout()  # Adjust layout to fit everything neatly
    output_file = 'static/similar_images.png'
    plt.savefig(output_file)
    plt.close()
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
            
            # Save images to file
            img_file = save_similar_images(filepath, top_indices, filename, similarities, df)
            
            # Prepare data for rendering
            similar_images = [(filename[idx], similarities[idx]) for idx in top_indices]
            
            return render_template('results.html', images=similar_images, img_file=img_file)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


