# Image Similarity Search Application

# Smart Fashion Recommender

A visual similarity search system for fashion images.  
Given a clothing photo, the model finds and displays visually similar items along with brand info, product name, and price — using pre-trained neural networks and cosine similarity.


##  Summary

- **Goal:** Enable fast and accurate search of similar clothing images based on content.
- **Input:** User-uploaded image.
- **Output:** Visually similar items + metadata (brand, name, price, link).
- **Backend:** Python + ResNet50/MobileNetV3 + Nearest Neighbors.
- **Frontend:** Jupyter/Web interface (prototype).


##  Dataset Collection

Fashion product images were collected using the official **retailed.io API**.  
The `products_info.csv` file contains:

-  Brand  
-  Product name  
-  Short description  
-  Price  
-  Product URL  

Each image is linked to its metadata to improve user experience in the final app.


##  Models Used

###  ResNet-50
- Deep CNN with 50 layers, using residual blocks to avoid vanishing gradients.
- Pre-trained on ImageNet.
- Offers **high accuracy** and robust feature extraction.
- Best suited for backend systems with **GPU** access.

###  MobileNetV3
- Lightweight CNN designed for **mobile and embedded** devices.
- Fast, efficient, small in size.
- Also pre-trained on ImageNet.
- Suitable for integration into **mobile applications**.


##  Feature Extraction

- Images are resized and passed through the selected model.
- The output feature vector (usually 512 or 1024-dim) is stored.
- Cosine similarity is used to compare new queries against the dataset.


##  Image Similarity Search

- User uploads an image.
- The app extracts features using the selected model.
- Finds **top N most similar images** via Nearest Neighbors with cosine distance.
- Returns results along with full product metadata.


## File Structure

### 📁 File Structure

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



