import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline, ViTImageProcessor
from instaloader import Instaloader, Post
import os
from PIL import Image
import pytesseract

class InstagramPostAnalyzer:
    def __init__(self):
        self.object_detector = YOLO('yolov8n.pt')
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        self.context_classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            image_processor=ViTImageProcessor.from_pretrained(
                "google/vit-base-patch16-224",
                use_fast=True
            )
        )
        self.loader = Instaloader()
        self.content_categories = [
            "Lifestyle & Personal", "Fashion & Style", "Food & Dining",
            "Travel & Adventure", "Health & Fitness", "Beauty & Wellness",
            "Photography", "Art & Design", "Nature & Outdoors",
            "Pets & Animals", "Tech & Gadgets", "Gaming & Esports",
            "Motivation & Inspirational", "Comedy & Humor", "Family & Parenting",
            "Social & Friendship", "Shopping & Retail", "Events & Occasions",
            "Automotive", "Education", "Finance", "Sports", "Entertainment",
            "Other / Misc"
        ]
        self.object_categories = [
            'person', 'animal', 'food', 'vehicle', 'car', 'clothing',
            'electronics', 'furniture', 'nature', 'building',
            'sports_equipment', 'book', 'phone', 'laptop', 'money'
        ]

    def download_post(self, post_url):
        try:
            shortcode = post_url.split('/')[-2]
            post = Post.from_shortcode(self.loader.context, shortcode)
            self.loader.download_post(post, target='temp_post')
            return True
        except Exception as e:
            print(f"Error downloading post: {e}")
            return False

    def analyze_image(self, image_path):
        image = cv2.imread(image_path)
        objects_detected = self.detect_objects(image)
        context = self.analyze_context(image)
        text, text_sentiment = self.analyze_text(image)
        return {
            'objects': objects_detected,
            'context': context,
            'text': text,
            'text_sentiment': text_sentiment
        }

    def detect_objects(self, image):
        results = self.object_detector(image)
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.object_detector.names[class_id]
                if confidence > 0.5:
                    detected_objects.append({
                        'label': label,
                        'confidence': confidence
                    })
        return detected_objects

    def analyze_context(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        predictions = self.context_classifier(pil_image)
        top_prediction = predictions[0]['label']
        confidence = predictions[0]['score']
        category_mapping = {
            'landscape': 'Nature & Outdoors',
            'dog': 'Pets & Animals',
            'cat': 'Pets & Animals',
            'person': 'Lifestyle & Personal',
            'meal': 'Food & Dining'
        }
        category = category_mapping.get(top_prediction.lower(), "Other / Misc")
        return {
            'category': category,
            'confidence': confidence,
            'raw_prediction': top_prediction
        }

    def analyze_text(self, image):
        text = pytesseract.image_to_string(image)
        if text.strip():
            sentiment = self.sentiment_analyzer(text)[0]
            return text, sentiment
        return None, None

    def categorize_post(self, analysis_results):
        category_scores = {cat: 0 for cat in self.content_categories}
        objects = [obj['label'].lower() for obj in analysis_results['objects']]
        for obj in objects:
            if 'food' in obj:
                category_scores['Food & Dining'] += 2
            if 'cloth' in obj:
                category_scores['Fashion & Style'] += 2
            if 'dog' in obj or 'cat' in obj or 'animal' in obj:
                category_scores['Pets & Animals'] += 2
            if 'car' in obj or 'vehicle' in obj:
                category_scores['Automotive'] += 2
            if 'sports_equipment' in obj:
                category_scores['Sports'] += 2
            if 'phone' in obj or 'laptop' in obj or 'electronics' in obj:
                category_scores['Tech & Gadgets'] += 2
            if 'book' in obj:
                category_scores['Education'] += 2
            if 'money' in obj:
                category_scores['Finance'] += 2
        context_category = analysis_results['context']['category']
        if context_category in category_scores:
            category_scores[context_category] += 3
        else:
            category_scores["Other / Misc"] += 1
        final_category = max(category_scores.items(), key=lambda x: x[1])[0]
        return final_category

    def analyze(self, post_url):
        if not self.download_post(post_url):
            return None
        image_path = None
        for file in os.listdir('temp_post'):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join('temp_post', file)
                break
        if not image_path:
            print("No image found in downloaded post.")
            for file in os.listdir('temp_post'):
                os.remove(os.path.join('temp_post', file))
            os.rmdir('temp_post')
            return None
        analysis_results = self.analyze_image(image_path)
        final_category = self.categorize_post(analysis_results)
        for file in os.listdir('temp_post'):
            os.remove(os.path.join('temp_post', file))
        os.rmdir('temp_post')
        return {
            'category': final_category,
            'objects_detected': analysis_results['objects'],
            'context': analysis_results['context'],
            'text': analysis_results['text'],
            'text_sentiment': analysis_results['text_sentiment']
        }

if __name__ == "__main__":
    analyzer = InstagramPostAnalyzer()
    #USE OTHER EXAMPLES
    post_url = "https://www.instagram.com/p/DG2yopigqIn/"
    results = analyzer.analyze(post_url)
    if results:
        print("=========================================")
        print("           ANALYSIS RESULTS             ")
        print("=========================================\n")
        print(f" Category            : {results['category']}")
        print(f" Objects Detected    :")
        for obj in results['objects_detected']:
            print(f"   - {obj['label']} (Confidence: {obj['confidence']:.2f})")
        print()
        print(f" Context             : {results['context']['category']} "
              f"(Confidence: {results['context']['confidence']:.2f}, "
              f"Raw: {results['context']['raw_prediction']})")
        print()
        if results['text']:
            print(f" Text Extracted      : {results['text'].strip()}")
            print(f" Text Sentiment      : {results['text_sentiment']['label']}"
                  f" (Score: {results['text_sentiment']['score']:.2f})")
        else:
            print(" No text detected from image.")
        print("\n=========================================")
    else:
        print("Analysis failed")
