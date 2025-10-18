"""
Sarcasm Detection Model Training Script
This script downloads a dataset, trains a machine learning model, and saves it for deployment.
"""

import pandas as pd
import numpy as np
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import string

# ============================================================================
# STEP 1: DOWNLOAD AND LOAD DATASET
# ============================================================================

def download_sarcasm_dataset():
    """
    Downloads the News Headlines Dataset for Sarcasm Detection from a public source.
    This dataset contains news headlines labeled as sarcastic or not.
    """
    print("Downloading sarcasm dataset...")
    
    # Using a well-known sarcasm detection dataset
    url = "/home/harsha/sarcasm_detection_project/Sarcasm_Headlines_Dataset.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open('sarcasm_data.json', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Creating a sample dataset instead...")
        create_sample_dataset()
        return False

def create_sample_dataset():
    """
    Creates a sample dataset if download fails.
    """
    sample_data = [
        {"headline": "Breaking: Local Man Discovers Water Is Wet", "is_sarcastic": 1},
        {"headline": "Study Finds That 100% Of People Who Breathe Air Eventually Die", "is_sarcastic": 1},
        {"headline": "Area Man Consults Internet Before Arguing With Wife", "is_sarcastic": 1},
        {"headline": "Scientists Discover New Species of Butterfly in Amazon", "is_sarcastic": 0},
        {"headline": "Local School Wins State Championship", "is_sarcastic": 0},
        {"headline": "New Study Shows Benefits of Regular Exercise", "is_sarcastic": 0},
        {"headline": "Company Announces Record Profits This Quarter", "is_sarcastic": 0},
        {"headline": "Groundbreaking: Man Puts Shopping Cart Back Where It Belongs", "is_sarcastic": 1},
        {"headline": "Area Woman Still Waiting For Perfect Time To Use Nice Candles", "is_sarcastic": 1},
        {"headline": "Study: People Who Say 'No Offense' About To Say Something Offensive", "is_sarcastic": 1},
    ] * 200  # Repeat to create a larger dataset
    
    with open('sarcasm_data.json', 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print("Sample dataset created!")

def load_dataset():
    """
    Loads the sarcasm dataset from JSON file.
    """
    data = []
    with open('sarcasm_data.json', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"\nDataset loaded: {len(df)} samples")
    print(f"Sarcastic: {df['is_sarcastic'].sum()}, Non-sarcastic: {len(df) - df['is_sarcastic'].sum()}")
    
    return df

# ============================================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================================

def clean_text(text):
    """
    Cleans and preprocesses text data.
    - Converts to lowercase
    - Removes URLs, mentions, hashtags
    - Removes punctuation and special characters
    - Removes extra whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_data(df):
    """
    Preprocesses the entire dataset.
    """
    print("\nPreprocessing data...")
    
    # Clean text
    df['cleaned_text'] = df['headline'].apply(clean_text)
    
    # Remove empty strings
    df = df[df['cleaned_text'].str.len() > 0]
    
    print(f"Preprocessing complete! {len(df)} samples remaining.")
    
    return df

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT AND VECTORIZATION
# ============================================================================

def prepare_features(df, test_size=0.2, random_state=42):
    """
    Splits data and creates TF-IDF features.
    """
    print("\nPreparing features...")
    
    # Split data
    X = df['cleaned_text']
    y = df['is_sarcastic']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # TF-IDF Vectorization - IMPROVED PARAMETERS
    # Using more features and better n-gram range
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased from 5000
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85,  # Changed from 0.9
        sublinear_tf=True  # Added for better feature scaling
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF features created: {X_train_tfidf.shape[1]} features")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, model_type='logistic'):
    """
    Trains a machine learning model for sarcasm detection.
    
    Args:
        model_type: 'logistic' or 'random_forest'
    """
    print(f"\nTraining {model_type} model...")
    
    if model_type == 'logistic':
        # IMPROVED PARAMETERS
        model = LogisticRegression(
            max_iter=2000,  # Increased from 1000
            C=0.5,  # Changed from 1.0 for better regularization
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Added to handle class imbalance
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=30,  # Increased from 20
            min_samples_split=5,  # Added
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Added
        )
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    return model

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance on test set.
    """
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

# ============================================================================
# STEP 6: SAVE MODEL AND VECTORIZER
# ============================================================================

def save_model(model, vectorizer, model_path='sarcasm_model.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Saves the trained model and vectorizer to disk.
    """
    print("\nSaving model and vectorizer...")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that runs the entire training pipeline.
    """
    print("=" * 80)
    print("SARCASM DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # Step 1: Download/Load dataset
    download_sarcasm_dataset()
    df = load_dataset()
    
    # Step 2: Preprocess
    df = preprocess_data(df)
    
    # Step 3: Prepare features
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(df)
    
    # Step 4: Train model (using Random Forest for better performance)
    model = train_model(X_train, y_train, model_type='random_forest')
    
    # Step 5: Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 6: Save model
    save_model(model, vectorizer)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("=" * 80)
    
    # Test with sample predictions
    print("\nSample Predictions:")
    test_samples = [
        "Scientists discover new planet in solar system",
        "Man finally uses turn signal, world celebrates",
        "Company announces new product launch",
        "Oh great another meeting that could have been an email",
        "Study finds that 100 percent of people who breathe air eventually die"
    ]
    
    for sample in test_samples:
        cleaned = clean_text(sample)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        label = "SARCASTIC" if prediction == 1 else "NOT SARCASTIC"
        confidence = probability[prediction] * 100
        
        print(f"\nText: {sample}")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()