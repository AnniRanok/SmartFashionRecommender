#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import cv2
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load precomputed features and filenames
with open('featurevector_resnet.pkl', 'rb') as f:
    feature_list = np.array(pickle.load(f))

with open('filenames_resnet.pkl', 'rb') as f:
    filename = pickle.load(f)

# Load model
model = load_model('resnet50_model.keras')

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

# This is the new function for displaying black images
def display_black_images(selected_image_path, top_indices, filename, similarities, df):
    fig, axes = plt.subplots(1, 6, figsize=(25, 5))
    fig.suptitle('Top 5 Similar Images')

    # Display the query image as a black square
    axes[0].imshow(np.zeros((224, 224, 3), dtype=np.uint8))  # Black image
    axes[0].set_title('Query')
    axes[0].axis('off')

    # Display black squares for similar images
    for i, idx in enumerate(top_indices):
        axes[i + 1].imshow(np.zeros((224, 224, 3), dtype=np.uint8))  # Black image
        axes[i + 1].set_title(f"Similarity: {similarities[idx]:.2f}")
        axes[i + 1].axis('off')

        # Display additional information
        img_name = os.path.basename(filename[idx])
        img_metadata = df[df['Image Name'] == img_name].iloc[0]
        brand = img_metadata['Brand']
        product_name = img_metadata['Product Name']
        product_info = f"{brand}\n{product_name}"
        axes[i + 1].text(0.5, -0.2, product_info, fontsize=10, ha='center', transform=axes[i + 1].transAxes)

    plt.show()

# Commented out the real function
# def display_similar_images(selected_image_path, top_indices, filename, similarities, df):
#     fig, axes = plt.subplots(1, 6, figsize=(25, 5))
#     fig.suptitle('Top 5 Similar Images')

#     # Display the query image
#     selected_img = cv2.imread(selected_image_path)
#     selected_img = cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB)
#     axes[0].imshow(selected_img)
#     axes[0].set_title('Query')
#     axes[0].axis('off')

#     # Display similar images
#     for i, idx in enumerate(top_indices):
#         img_path = filename[idx]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         axes[i + 1].imshow(img)
#         axes[i + 1].set_title(f"Similarity: {similarities[idx]:.2f}")
#         axes[i + 1].axis('off')

#         # Display additional information
#         img_name = os.path.basename(img_path)
#         img_metadata = df[df['Image Name'] == img_name].iloc[0]
#         brand = img_metadata['Brand']
#         product_name = img_metadata['Product Name']
#         product_info = f"{brand}\n{product_name}"
#         axes[i + 1].text(0.5, -0.2, product_info, fontsize=10, ha='center', transform=axes[i + 1].transAxes)

#     plt.show()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            
            # Extract features and recommend similar images
            top_indices, similarities = recommend_similar_images(filepath, model, feature_list, filename)
            
            # Prepare data for rendering
            similar_images = [(filename[idx], similarities[idx]) for idx in top_indices]
            
            # Use the new function for testing
            display_black_images(filepath, top_indices, filename, similarities, df)
            
            return render_template('results.html', images=similar_images)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)


