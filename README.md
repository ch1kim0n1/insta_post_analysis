# Instagram Post Analyzer (AI Generated)

## Overview

This project analyzes Instagram posts using multiple AI components. It detects objects in images with YOLO, classifies image context with a ViT-based model, extracts any text found in the image with Tesseract OCR, and performs sentiment analysis on that text. It also categorizes the post into one of several predefined content categories.

## Features

1. **Object Detection**:
   - Utilizes YOLO (yolov8n.pt) to identify objects in the image.
2. **Context Classification**:
   - Uses a ViT (Vision Transformer) pipeline to classify the overall context or scene.
3. **Optical Character Recognition (OCR)**:
   - Extracts text from images using Tesseract.
4. **Sentiment Analysis**:
   - Determines sentiment from extracted text with a DistilBERT-based model.
5. **Categorization**:
   - Categorizes the post into one of several content categories (e.g., "Food & Dining," "Tech & Gadgets," "Sports," etc.).
6. **Automated Download**:
   - Retrieves Instagram posts automatically via Instaloader.

## Requirements

- Python 3.7 or higher
- OpenCV
- NumPy
- ultralytics (for YOLO)
- transformers
- instaloader
- requests
- Pillow (PIL)
- pytesseract
- Tesseract OCR installed on your system

## Installation

1. Clone or download the project to your local system.
2. Create and activate a virtual environment (recommended).
3. Install the required packages:
