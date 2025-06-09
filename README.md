# EEG-Based Emotion Classification Using LSTM
This project investigates the use of EEG (electroencephalogram) signals to classify human emotional states—**negative**, **neutral**, and **positive**—using a deep learning approach based on Long Short-Term Memory (LSTM) networks. The project explores how such classification could theoretically inform adaptive user interface (UI) designs, although no actual UI is implemented.

## Project Summary
- Uses EEG data to recognize emotional states based on brain activity.
- Implements an LSTM network due to its effectiveness in handling sequential data like EEG signals.
- Analyzes data from controlled experiments using emotion-inducing video stimuli.

## Dataset
The dataset is based on public EEG recordings where subjects watched emotion-inducing video clips:
- 3 emotion classes: **Positive**, **Neutral**, **Negative**
- Features include statistical, frequency-based, and entropy measures
- Preprocessing includes normalization, feature selection, and dimensionality reduction (e.g., t-SNE)

## Methodology
- **Model**: Two-layer LSTM with dropout, ReLU activation, and softmax output
- **Training**: 80/20 train-test split with K-Fold cross-validation and class weighting
- **Metrics**: Accuracy, F1-score, Confusion Matrix

## Results
- LSTM alone achieved performance comparable to CNN-LSTM hybrid models
- Confusion matrices and classification reports showed effective emotion separation
- Clear emotion clusters observed in t-SNE plots and correlation heatmaps

