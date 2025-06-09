import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

# Load and Prepare Data
from google.colab import drive
drive.mount('/content/drive')
def load_data(file_path):
    data = pd.read_csv(file_path)
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    data['label'] = data['label'].map(label_mapping)
    return data
file_path = '/content/drive/MyDrive/emotions.csv'
data = load_data(file_path)

# Data Preprocessing
def preprocess_data(data):
    # Separate features and labels
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape data for LSTM
    timesteps = 182  # Based on divisors
    num_features = X_scaled.shape[1] // timesteps
    X_reshaped = X_scaled.reshape(-1, timesteps, num_features)

    print(f"Data shape after preprocessing: {X_reshaped.shape}")
    return X_reshaped, y, scaler

# LSTM Model
def create_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create Callbacks
def create_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        )
    ]

# Training with K-fold Cross-validation
def train_with_kfold(X, y, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    best_model = None
    best_score = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start_time = time.time()

        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_fold),
            y=y_train_fold
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Create and train model
        model = create_model((X.shape[1], X.shape[2]))

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=30,
            batch_size=64,
            class_weight=class_weight_dict,
            callbacks=create_callbacks(),
            verbose=1
        )

        # Evaluate
        score = model.evaluate(X_val_fold, y_val_fold)[1]
        fold_scores.append(score)

        # Keep track of best model
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, fold_scores

# Evaluation of the Model
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating final model...")
    # Make predictions
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest Accuracy: {test_accuracy:.3f}')

    # Print classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred_classes))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Preprocess data
X, y, scaler = preprocess_data(data)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with k-fold cross-validation
best_model, fold_scores = train_with_kfold(X_train, y_train)

# Evaluate the final model
evaluate_model(best_model, X_test, y_test)

import random

# Define emotion labels dictionary
emotion_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

# Get random predictions
for i in range(6):
    # Generate random index
    random_index = random.randint(0, len(X_test) - 1)

    # Get sample and reshape for LSTM input
    sample_input = X_test[random_index:random_index+1]  # Using slice to keep dimensions
    true_label = y_test[random_index]
    true_emotion = emotion_labels.get(true_label, 'Unknown')

    # Make prediction
    predicted_probs = best_model.predict(sample_input, verbose=0)
    predicted_label = np.argmax(predicted_probs)
    predicted_emotion = emotion_labels.get(predicted_label, 'Unknown')
    confidence = predicted_probs[0][predicted_label] * 100

    # Display results
    print(f"\nSample {i+1}:")
    print(f"Real Emotion: {true_emotion}")
    print(f"Predicted Emotion: {predicted_emotion}")
    print(f"Confidence: {confidence:.2f}%")
