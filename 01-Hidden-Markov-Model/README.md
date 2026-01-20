# HMM-Based Digit Recognition System

## Overview

This project implements a **Hidden Markov Model (HMM)-based digit recognition system** that uses audio features (MFCC) to recognize spoken digits (0-9). It's a practical application of HMMs in speech processing and pattern recognition.

## What is a Hidden Markov Model?

### Concept
A Hidden Markov Model is a statistical model representing a system with the following properties:

- **Hidden States**: The system has internal states that are not directly observable
- **Markov Property**: Future states depend only on the current state, not on the history
- **Probabilistic Transitions**: Transitions between states occur with certain probabilities
- **Emissions**: Each state produces observable outputs (emissions) with certain probabilities

### How It Works for Digit Recognition
1. Each digit (0-9) has its own HMM with 3 hidden states
2. During training, the model learns the probability distributions of transitions between states
3. Each state learns to emit MFCC feature vectors characteristic of that digit
4. During recognition, the model calculates the likelihood of the audio belonging to each digit
5. The digit with the highest likelihood is the recognized digit

### Mathematical Foundation
The key computation is the likelihood: P(features | model) calculated using the forward algorithm.

## Project Structure

```
Hidden-Markov-Model/
├── HMM.py              # Main implementation with fixed bugs
├── demo.py             # Complete demo with training and testing
├── generate_samples.py # Script to generate synthetic audio samples
├── audio_samples/      # Directory containing training data (auto-generated)
│   ├── 0/
│   ├── 1/
│   └── ...
│   └── 9/
└── README.md          # This file
```

## Features

- **MFCC Feature Extraction**: Uses librosa to extract Mel-Frequency Cepstral Coefficients (13 coefficients)
- **Gaussian HMM**: Implements Gaussian HMM with 3 hidden states per digit
- **Multiple Training Samples**: Each digit trained on 5 synthetic variations
- **Synthetic Audio Generation**: Creates realistic synthetic audio for training
- **Interactive Testing**: Tests on generated samples and demonstrates recognition

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Create and activate virtual environment** (if not already done):
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy: Numerical computing
- librosa: Audio processing and feature extraction
- hmmlearn: Hidden Markov Model implementation
- soundfile: Audio file I/O
- scikit-learn: Machine learning utilities (dependency)

## Usage

### Running the Complete Demo

The demo script performs all steps automatically:

```bash
cd Hidden-Markov-Model
python demo.py
```

**What the demo does:**
1. ✓ Generates 50 synthetic audio samples (5 per digit)
2. ✓ Trains 10 Gaussian HMM models (one per digit)
3. ✓ Tests recognition on training samples
4. ✓ Shows accuracy metrics
5. ✓ Performs interactive test on a synthesized digit 5 audio
6. ✓ Displays top 3 predictions with confidence scores

### Using Individual Components

#### 1. Generate Training Data
```bash
python generate_samples.py
```
Creates `./audio_samples/` directory with 50 synthetic audio files.

#### 2. Train Models (from HMM.py)
```python
from HMM import train_hmm_models
digit_models = train_hmm_models()
```

#### 3. Recognize a Digit
```python
from HMM import recognize_digit
recognized_digit = recognize_digit('path/to/audio.wav')
print(f"Recognized digit: {recognized_digit}")
```

## Code Explanation

### MFCC Feature Extraction
```python
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T  # Transpose to (n_frames, 13)
```
- Loads audio file and extracts MFCC coefficients
- MFCCs capture the perceptual characteristics of sound
- Returns shape: (time_frames, 13 features)

### HMM Training
```python
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
X = np.concatenate(samples)  # Combine all sample features
lengths = [sample.shape[0] for sample in samples]  # Frame counts
model.fit(X, lengths=lengths)
```
- `n_components=3`: Each digit has 3 hidden states
- `covariance_type="diag"`: Diagonal covariance (features are conditionally independent)
- `n_iter=100`: Maximum 100 EM iterations
- `lengths`: Specifies boundaries between different training sequences

### Digit Recognition
```python
scores = {digit: model.score(features) for digit, model in digit_models.items()}
recognized_digit = max(scores, key=scores.get)
```
- Calculates log-likelihood score for each digit model
- Selects digit with highest score
- Higher score = better match to the digit's HMM pattern

## Key Improvements Made

### Bug Fixes in Original Code
1. **Fixed method name**: `model.scores()` → `model.score()` (correct hmmlearn API)
2. **Fixed return statement**: Was returning function instead of result
3. **Fixed lengths calculation**: Used `sample.shape[0]` instead of `len(samples)`
4. **Fixed variable name typo**: `recingized_digit` → `recognized_digit`

### Enhancements
- Comprehensive error handling
- Detailed progress logging
- Synthetic audio generation with realistic variations
- Multiple test samples for robustness
- Confidence scores and top-3 predictions display
- Training/testing phase separation

## Model Parameters

Current configuration:
- **HMM Components**: 3 hidden states per digit
- **Covariance Type**: Diagonal (conditional independence of features)
- **Training Iterations**: 100 EM iterations
- **MFCC Features**: 13 coefficients
- **Samples per Digit**: 5 training samples

### Tuning Parameters
To improve performance, you can adjust:
- `n_components`: Try 2, 3, 4, or 5 states
- `n_mfcc`: Try 13, 20, or 26 coefficients
- `n_iter`: Increase to 200-500 for better convergence
- Number of training samples: More samples = better generalization

## Performance Notes

### Synthetic vs. Real Audio
- Current system uses **synthetic audio** for demonstration
- For real-world digit recognition, train on actual spoken digit recordings
- Synthetic audio ensures reproducibility but doesn't capture speech variability

### Expected Accuracy
- **On synthetic data**: 90-100% (well-separated frequency patterns)
- **On real speech**: 70-95% (depends on speaker variation, noise, accent)

### Computational Complexity
- **Training**: O(n_samples × n_frames × n_components²)
- **Recognition**: O(n_frames × n_components²)
- Very fast on modern hardware

## Real-World Applications

1. **Voice Control Systems**: Command recognition, virtual assistants
2. **Speech Recognition**: Phoneme recognition, word spotting
3. **Gesture Recognition**: Temporal pattern matching
4. **Bioinformatics**: Protein sequence analysis
5. **Financial Analysis**: Time series pattern detection
6. **Video Analysis**: Action recognition from motion features

## Troubleshooting

### Issue: "No module named 'librosa'"
**Solution**: Install with `pip install librosa`

### Issue: "No module named 'hmmlearn'"
**Solution**: Install with `pip install hmmlearn`

### Issue: "RuntimeError: converged flag not found"
**Solution**: Update hmmlearn: `pip install --upgrade hmmlearn`

### Issue: Low accuracy
**Try**:
- Increase `n_components` to 4 or 5
- Add more training samples per digit
- Use real audio instead of synthetic audio
- Increase `n_mfcc` to 20 or higher

### Issue: Audio file not found
**Make sure**:
1. Run `generate_samples.py` first to create training data
2. Or provide correct path to your audio files
3. Check that files have `.wav` extension

## Dependencies and Versions

See `requirements.txt` for complete list. Key dependencies:

```
numpy>=1.20.0          # Numerical arrays
librosa>=0.10.0        # Audio processing
hmmlearn>=0.3.0        # Hidden Markov Models
scikit-learn>=1.0.0    # Machine learning utilities
soundfile>=0.12.0      # Audio file I/O
```

## Further Learning

### Concepts to Explore
1. **Viterbi Algorithm**: Finding most likely state sequence
2. **Forward-Backward Algorithm**: Computing probabilities efficiently
3. **Baum-Welch Algorithm**: EM-based model training
4. **MFCCs**: Perceptual frequency representation
5. **Dynamic Time Warping**: Alternative temporal pattern matching

### Related Methods
- **GMM (Gaussian Mixture Models)**: For acoustic modeling
- **Deep Learning**: LSTM/CNN for sequence recognition
- **DTW (Dynamic Time Warping)**: Template matching
- **Neural Networks**: End-to-end learning

### References
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)
- [Librosa Documentation](https://librosa.org/)
- [Hidden Markov Models Tutorial](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications"

## License

This project is provided for educational purposes.

## Author

Created for NLP course (CSE 3523)

## Questions or Issues?

Refer to the code comments or check the troubleshooting section above.

---

**Happy digit recognition! 🎙️🔢**
