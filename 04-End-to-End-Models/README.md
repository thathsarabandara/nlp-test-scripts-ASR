# End-to-End CNN+LSTM Model for Automatic Speech Recognition (ASR)

## Overview

This project implements an **End-to-End CNN+LSTM model** for Automatic Speech Recognition. Unlike traditional ASR pipelines that separate feature extraction, acoustic modeling, and language modeling, end-to-end models learn all aspects jointly in one unified architecture.

The key innovation is combining:
- **CNN (Convolutional Neural Network)**: Extracts spatial-temporal features from spectrograms
- **LSTM (Long Short-Term Memory)**: Models temporal dependencies in the extracted features
- **CTC Loss**: Handles variable-length input-output alignment automatically

This architecture is more powerful than pure LSTM-based models because CNNs excel at capturing local patterns in audio spectrograms.

## Why End-to-End Models?

### Traditional ASR Pipeline
```
Audio → Feature Extraction → Acoustic Model → Language Model → Text
```
**Problems:**
- Multiple separate systems to train and maintain
- Errors compound through the pipeline
- Manually designed features may not be optimal

### End-to-End Approach
```
Audio → Joint CNN+LSTM+CTC Model → Text
```
**Advantages:**
- Single unified model trained on all tasks jointly
- No compounding errors
- Model learns optimal features automatically
- Better performance with sufficient training data

## Architecture Deep Dive

### 1. Input Layer
```python
input_data = Input(shape=(None, 13), name='audio_input')
```

**Shape Explanation:**
- `(None, 13)` means variable number of time steps with 13 MFCC features
- Each audio file has different duration → different number of time steps
- 13 MFCC coefficients capture spectral characteristics of speech

**Example Input Dimension:**
- Audio file 1: (500, 13) → 500 frames, 13 features
- Audio file 2: (750, 13) → 750 frames, 13 features
- Padding makes them same size for batch processing

### 2. CNN Feature Extraction Layer

```python
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), 
                    activation='relu', padding='same')(input_data)
```

**Why CNN for Speech?**

Speech spectrograms have local structure:
- **Vertical patterns**: Consecutive time frames are similar (temporal smoothness)
- **Horizontal patterns**: Adjacent frequency bands are correlated (frequency continuity)
- **Diagonal patterns**: Formant transitions (frequency changes over time)

CNNs capture these patterns through convolutional filters.

**Parameter Breakdown:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `filters=32` | 32 | Learn 32 different feature maps (optimal balance) |
| `kernel_size=(3,3)` | 3×3 | Small receptive field captures local patterns |
| `activation='relu'` | ReLU | Non-linearity enables learning complex features |
| `padding='same'` | Same | Preserve spatial dimensions for feature maps |

**Why these choices?**
- **32 filters**: Not too few (underfitting) or too many (overfitting, slow)
- **3×3 kernel**: Captures adjacent frames and neighboring frequencies
- **ReLU**: Computationally efficient, handles vanishing gradient
- **Same padding**: Output size = input size, no information loss at boundaries

**Output Shape:**
- Input: (batch, time_steps, 13, 1) — *added channel dimension*
- Conv2D output: (batch, time_steps, 13, 32) — *32 feature maps*

### 3. MaxPooling Layer

```python
pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
```

**What it does:**
- Takes maximum value in each 2×2 window
- Reduces spatial dimensions by half
- Keeps most important features, discards less important

**Why MaxPooling?**

1. **Dimensionality Reduction**: Fewer parameters to process
2. **Shift Invariance**: Model robust to slight shifts in features
3. **Compute Efficiency**: Faster training and inference
4. **Overfitting Prevention**: Acts as regularization

**Output Shape:**
- MaxPool output: (batch, time_steps//2, 13//2, 32)
- Example: (32, 500, 13, 32) → (32, 250, 6, 32)

**Trade-off Explanation:**
- We lose some information (half the spatial resolution)
- But gain significant computational speedup
- In speech, this trade-off is favorable—high-frequency details aren't critical

### 4. Flatten Layer

```python
flatten_layer = Flatten()(pool_layer)
```

**Purpose:** Convert 3D feature maps to 1D vector for LSTM processing

**Shape Transformation:**
```
(batch, 250, 6, 32) → (batch, 250, 6*32) → (batch, 250, 192)
```

**Why flatten?**
- LSTM expects 3D input: (batch, time_steps, features)
- After flattening, each time step has 192 features (6×32)
- Now LSTM can process the extracted CNN features

### 5. LSTM Layer

```python
rnn_layer = LSTM(units=128, return_sequences=True)(flatten_layer)
```

**What LSTM does:**
- Processes variable-length sequences
- Maintains hidden state across time steps
- Learns long-term temporal dependencies
- Handles vanishing gradient problem (compared to basic RNN)

**Parameter Details:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `units=128` | 128 | Hidden state size and output dimension |
| `return_sequences=True` | True | Output all time steps (needed for CTC) |

**Why 128 units?**
- **Too few (<64)**: Model can't capture complex dependencies → underfitting
- **Too many (>256)**: Overfitting, slower training, more memory
- **128**: Sweet spot for medium-complexity speech patterns

**return_sequences=True Explanation:**

Without this flag, LSTM returns only the last time step:
```
Input:  (batch, 250, 192)
Output: (batch, 128)  ← only last time step
```

With `return_sequences=True`:
```
Input:  (batch, 250, 192)
Output: (batch, 250, 128)  ← all time steps preserved
```

**Why keep all time steps?**
- CTC needs predictions for every time step
- We want character predictions for each frame
- Each time step will eventually map to a character or blank token

### 6. TimeDistributed Dense Layer

```python
logits = TimeDistributed(Dense(num_chars + 1, activation='softmax'))(rnn_layer)
```

**What TimeDistributed does:**
- Applies the same Dense layer to each time step independently
- Converts LSTM output (128 dims) to character probabilities (28 dims)

**Shape Transformation:**
```
Input (from LSTM):  (batch, 250, 128)
Output:             (batch, 250, 28)
```

Each of the 250 time steps gets a 28-dimensional probability distribution.

**Layer Details:**

```python
Dense(num_chars + 1, activation='softmax')
```

- **num_chars + 1 = 28**:
  - 27 characters (a-z + space)
  - +1 for CTC "blank" token (internal use)

- **softmax activation**:
  - Converts outputs to probability distribution
  - Sum of probabilities = 1.0
  - Example: [0.05, 0.15, 0.70, 0.10, ...] for "cat"

**Why TimeDistributed?**

Without it, we'd apply Dense only once (losing per-step predictions).
With it, each time step independently predicts a character.

## Model Compilation

```python
model.compile(optimizer=Adam(), loss=CTCLoss(), metrics=['accuracy'])
```

### Components:

**Optimizer: Adam**
- Adaptive Moment Estimation
- Maintains per-parameter learning rates
- Faster convergence than SGD
- Good default for most tasks

**Loss Function: CTCLoss**
- Handles variable-length sequences
- No need for frame-level alignment
- Automatically finds best alignment
- Sums probability over all possible alignments

**Metrics: Accuracy**
- Tracks training progress
- Shows how well model predicts characters

## Data Generation and Training

### Data Generator
```python
def data_generator(audio_paths, transcriptions, char_map, batch_size=32):
    while True:
        # Select random batch
        batch_indices = np.random.choice(len(audio_paths), size=batch_size)
        
        # Preprocess: audio → MFCC, text → indices
        batch_audio = [preprocess_audio(audio_paths[i]) for i in batch_indices]
        batch_transcriptions = [preprocess_transcription(transcriptions[i], 
                                                          char_map) 
                                for i in batch_indices]
        
        # Pad sequences to same length
        padded_audio = pad_sequences(batch_audio, padding='post')
        padded_transcriptions = pad_sequences(batch_transcriptions, padding='post')
        
        yield inputs, outputs
```

**Why a generator?**
1. **Memory efficiency**: Don't load entire dataset
2. **Infinite supply**: Can train multiple epochs
3. **Flexibility**: Apply data augmentation dynamically

**batch_size=32 Reasoning:**

| Batch Size | Pros | Cons |
|-----------|------|------|
| 8 | More updates per epoch | Noisier gradients, less GPU utilization |
| 32 | Good balance | Moderate memory usage |
| 128 | Smooth gradients, GPU efficient | More memory, fewer updates |
| 256 | Maximum efficiency | May not fit in memory |

32 is industry standard for speech recognition.

## Training Process

```python
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=val_generator,
    validation_steps=validation_steps
)
```

**Parameters:**
- **steps_per_epoch**: How many batches per epoch
  - Example: 1000 samples / 32 batch_size = ~31 steps
- **epochs=10**: Training iterations through full dataset
  - More epochs = longer training but potential overfitting
  - Monitor validation loss to avoid overfitting
- **validation_data**: Separate dataset to prevent overfitting

## Inference and Decoding

```python
example_audio = preprocess_audio(example_audio_path)
example_audio = np.expand_dims(example_audio, axis=0)  # Add batch dimension
prediction = model.predict(example_audio)

decoded_prediction = K.ctc_decode(
    prediction,
    input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
    greedy=True
)[0][0]
```

### Prediction Process:

1. **Preprocess**: Extract MFCC features
2. **Add batch dimension**: (time_steps, 13) → (1, time_steps, 13)
3. **Model prediction**: (1, time_steps, 28) — probabilities per character per frame
4. **CTC Decode**: Convert to text

### CTC Decoding Options:

**Greedy Decoding (greedy=True):**
```
At each time step, pick highest probability character
Advantages: Fast (O(n) complexity)
Disadvantages: Suboptimal (doesn't consider global context)
```

**Beam Search (greedy=False):**
```
Explore multiple paths, keep top-k probable sequences
Advantages: Better accuracy (considers context)
Disadvantages: Slower (O(n*beam_width) complexity)
```

## Advantages of CNN+LSTM Over Pure LSTM

### Feature Extraction Benefits:
| Aspect | Pure LSTM | CNN+LSTM |
|--------|-----------|----------|
| Learns from raw features | Requires perfect MFCC tuning | Learns optimal features |
| Local pattern recognition | Weak (looks at all features equally) | Strong (detects formants, patterns) |
| Parameter efficiency | Many parameters needed | Fewer parameters, better generalization |
| Computational cost | Moderate | Lower (CNN reduces dimensionality) |
| Performance on clean audio | Good | Better |
| Robustness to noise | Moderate | Better (CNN acts as feature filter) |

## Parameter Justification Summary

| Component | Parameter | Value | Justification |
|-----------|-----------|-------|---------------|
| Audio preprocessing | Sample rate | 16 kHz | Speech bandwidth ≈ 8 kHz |
| Audio preprocessing | MFCC coefficients | 13 | Perceptual relevance, standard choice |
| Conv2D | Filters | 32 | Balance: capacity vs efficiency |
| Conv2D | Kernel size | 3×3 | Captures local patterns |
| Conv2D | Activation | ReLU | Non-linearity, computational efficiency |
| MaxPooling | Pool size | 2×2 | 2× dimension reduction, preserve info |
| LSTM | Units | 128 | Medium complexity, fast training |
| LSTM | Return sequences | True | Need per-frame predictions for CTC |
| Dense | Output units | 28 | 27 chars + 1 blank token |
| Dense | Activation | Softmax | Probability distribution |
| Batch size | Size | 32 | Memory efficiency vs gradient quality |
| Training | Epochs | 10 | Usually enough for convergence |
| Optimizer | Type | Adam | Adaptive learning rate, stable |

## When to Use This Architecture

### ✅ Best For:
- Clean, well-recorded audio
- English or phonetic languages
- Medium vocabulary (< 100k words)
- GPU-available systems
- Quick prototyping

### ⚠️ Less Ideal For:
- Very noisy audio (need more data augmentation)
- Agglutinative languages (morphologically complex)
- Very large vocabulary (need language model)
- CPU-only systems (slow)
- Real-time inference (may be too slow)

## Potential Improvements

1. **Bidirectional LSTM**: Look at context from both directions
   ```python
   rnn_layer = Bidirectional(LSTM(64, return_sequences=True))(flatten_layer)
   ```

2. **Multiple CNN layers**: Deeper feature extraction
   ```python
   conv1 = Conv2D(32, (3,3), activation='relu')(input_data)
   conv2 = Conv2D(64, (3,3), activation='relu')(conv1)
   pool = MaxPooling2D((2,2))(conv2)
   ```

3. **Dropout**: Regularization to prevent overfitting
   ```python
   dropout = Dropout(0.3)(flatten_layer)
   ```

4. **Attention mechanism**: Focus on relevant frames
   ```python
   # Attention layers to select important features
   ```

5. **Language model**: Post-process predictions with n-grams
   ```python
   # Combine acoustic model with language model
   ```

6. **Beam search decoding**: More accurate than greedy
   - Use `greedy=False` in CTC decode

7. **Data augmentation**: Improve robustness
   - Add background noise
   - Time-stretch audio
   - Pitch shift

## References

- **Original E2E ASR Paper**: Graves et al. "Towards End-to-End Speech Recognition with Recurrent Neural Networks" (2014)
- **CTC Loss**: Graves et al. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks" (2006)
- **LSTM**: Hochreiter & Schmidhuber "Long Short-Term Memory" (1997)
- **CNN for Speech**: Abdel-Hamid et al. "Convolutional Neural Networks for Speech Recognition" (2014)

## Dependencies

```
numpy
matplotlib
librosa
tensorflow
keras
```

## Quick Start

1. **Prepare data:**
   ```python
   audio_paths = ['path/to/audio1.wav', 'path/to/audio2.wav', ...]
   transcriptions = ['hello world', 'goodbye', ...]
   ```

2. **Train model:**
   ```bash
   python ETE.py
   ```

3. **Use for prediction:**
   ```python
   model.load_weights('trained_model.h5')
   prediction = model.predict(preprocessed_audio)
   ```

## Performance Tips

- Use GPU for training (10-100× faster)
- Start with smaller dataset, verify convergence
- Monitor validation loss—stop if increasing
- Use data augmentation for small datasets
- Consider pre-training on large corpus
