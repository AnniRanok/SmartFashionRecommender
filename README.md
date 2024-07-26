# Image Similarity Search Application

### Summary of Sections:

- **Overview:** Brief description of what the application does.
- **Features:** Lists the main features.
- **Getting Started:** Instructions for setting up and running the application.
- **Usage:** How to interact with the web interface.
- **Troubleshooting:** Common issues and solutions.
- **Deployment on AWS:** Steps for deploying the application on AWS EC2.

## Overview

This application provides a web-based interface for finding similar images using a pre-trained ResNet50 model & MobileNetV3. 
Users can upload an image, and the application will return the most similar images from the database along with their similarity scores.

## Features

- Upload and analyze images
- Find and display similar images based on content
- View image metadata

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Flask
- TensorFlow
- OpenCV
- scikit-learn
- pandas

### Installation

1. **Set up the environment:**

   Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


2. **Install the required dependencies:**
  pip install -r requirements.txt

3. **Run the application locally:**
  python core-model-app/app1.py
  
    
## Usage

1. **Upload an image** using the web interface.
2. **View similar images** and their similarity scores.

## Troubleshooting

- **File not found**: Verify file locations and paths.
- **ModuleNotFoundError**: Ensure all dependencies are installed.


## Deployment on AWS

### Set Up an EC2 Instance:
Follow AWS documentation to launch an EC2 instance with your preferred operating system.

### Connect to the Instance:
ssh -i "your-key.pem" ubuntu@your-ec2-public-dns

### Install Required Packages and Clone the Repository:

- Update package lists: sudo apt-get update
- Install Python and other necessary packages:
- Clone the repository
- Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate
- Install the required dependencies:pip install -r requirements.txt

### Run the Application:
python app1.py

### Configure Security Groups:
Ensure that the security group associated with your EC2 instance allows inbound traffic on port 5000.

### Access the Application:
Open a web browser and navigate to http://your-ec2-public-dns:5000/.
