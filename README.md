# Smart Fashion Recommender (Image Similarity Search System)

## Overview

This project implements an image-based similarity search system for fashion products.

Given a query image, the system retrieves visually similar items from a product dataset using deep feature embeddings and cosine similarity.

The goal is to demonstrate a content-based image retrieval (CBIR) pipeline using pre-trained convolutional neural networks.


## Problem Statement

Traditional keyword-based search is limited for fashion e-commerce.

This system explores a visual search approach where similarity is computed directly from image embeddings rather than textual metadata.

## System Pipeline

The application follows a standard image retrieval pipeline:

1. Input image upload  
2. Feature extraction using a pre-trained CNN  
3. Embedding storage for dataset images  
4. Similarity computation using cosine distance  
5. Retrieval of top-N nearest items  
6. Metadata enrichment (brand, price, product info)


## Models

### ResNet-50
- Deep convolutional neural network with residual connections  
- Pre-trained on ImageNet  
- Used for high-quality feature extraction  

### MobileNetV3
- Lightweight CNN optimized for efficiency  
- Suitable for fast inference and constrained environments  
- Used for comparative experimentation  


## Feature Extraction

- Images are resized and normalized  
- Deep feature vectors are extracted from the selected model  
- Feature embeddings are stored as serialized vectors  
- Cosine similarity is used to compute nearest neighbors  


## Data

The dataset consists of fashion product images linked with metadata:

- Brand  
- Product name  
- Description  
- Price  
- Product URL  

Metadata is used for result enrichment in the retrieval interface.


## System Architecture

```plaintext
SmartFashionRecommender/
├── app1.py                     # Main application script
├── models/
│   └── yolov8n.pt              # (Optional) object detection model
├── featurevector_resnet.pkl    # Feature vectors (ResNet)
├── filenames_resnet.pkl        # Corresponding filenames
├── templates/
│   └── index.html              # Basic frontend (Jinja2)
├── requirements.txt            # Python dependencies
├── products_info.csv           # Product metadata (brand, price, URL)
├── test1.jpg                   # Sample image
├── test3.jpg                   # Sample image
└── README.md                   # Project documentation



## Output

For a given query image, the system returns:

- Top-N visually similar images  
- Associated product metadata  
- Brand and price information  
- Product reference links  


## Limitations

- Not trained end-to-end (uses pre-trained models only)  
- Dataset size is limited  
- No real-time model retraining pipeline  
- Prototype-level web interface  


## Tech Stack

- Python  
- TensorFlow / PyTorch (feature extraction)  
- ResNet-50 / MobileNetV3  
- Scikit-learn (Nearest Neighbors)  
- NumPy / Pandas  
- Flask / Jinja2 (prototype UI)  


## Status

This project is a computer vision prototype for content-based image retrieval in fashion recommendation systems.


## Key Focus Areas

- Deep learning feature extraction  
- Image embedding pipelines  
- Similarity search systems  
- Metadata enrichment for retrieval systems  
