# Connectionist Temporal Classification (CTC) for Automatic Speech Recognition (ASR)

## Overview

This project implements a **Connectionist Temporal Classification (CTC)** model for Automatic Speech Recognition (ASR). CTC is a specialized loss function designed for sequence-to-sequence problems where the alignment between input and output sequences is unknown. This is particularly useful for speech recognition where we don't have pre-aligned audio frames and transcriptions.

## What is CTC?

**Connectionist Temporal Classification** solves a fundamental problem in sequence-to-sequence modeling: when we don't know the alignment between input and output sequences.

### The Problem
- Audio input has variable length time steps
- Text transcription has variable length (characters)
- We don't know which audio frame corresponds to which character
- Traditional alignment methods require labeled character-level timing information

### The CTC Solution
CTC automatically learns the alignment by:
1. **Summing over all possible alignments** between input and output
2. **Including a "blank" token** to handle multiple frames per character
3. **Computing a loss function** that doesn't require alignment annotations
4. **Allowing many-to-one mappings** (e.g., multiple frames → one character)

### Key Advantage
No need for frame-level labels or pre-aligned data! CTC only requires:
- Audio file (input)
- Complete text transcription (output)

## Code Structure and Explanation

### 1. Imports
```python
import tensorflow as tf
import librosa
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras.losses import CTCLoss
```

**Why these libraries?**
- **TensorFlow/Keras**: Deep learning framework for building neural networks
- **librosa**: Audio processing library for extracting features from audio files
- **LSTM**: Recurrent neural network layer (captures temporal dependencies)
- **TimeDistributed**: Applies the same layer to each time step
- **CTCLoss**: The specialized loss function for CTC training

### 2. Data Preprocessing Functions

#### `preprocess_audio(audio_path)`
```python
def preprocess_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    return mfcc.T
```

**What it does:**
1. **Load audio**: Reads the audio file and resamples to 16,000 Hz (standard for speech)
2. **Extract MFCC features**: Converts raw audio to Mel-Frequency Cepstral Coefficients
3. **Transpose**: Returns shape `(time_steps, 13)` where 13 is number of MFCC coefficients

**Why MFCC?**
- Humans perceive sound on a logarithmic scale (Mel scale)
- MFCC captures perceptually relevant features
- Reduces dimensionality (audio sample rate 16,000 Hz → 13 coefficients)
- Standard for speech recognition tasks

**Parameter Details:**
- `sr=16000`: Sample rate (16kHz is standard for speech)
- `n_mfcc=13`: Number of MFCC coefficients (captures spectral information)
  - More coefficients = more detail but higher computation
  - 13 is a good balance between accuracy and efficiency

#### `preprocess_transcription(transcription, char_map)`
```python
def preprocess_transcription(transcription, char_map):
    return [char_map[char] for char in transcription]
```

**What it does:**
- Converts text transcription to numeric indices using a character map
- Example: "hello" → [8, 5, 12, 12, 15] (if 'h'=8, 'e'=5, etc.)

**Why numeric format?**
- Neural networks work with numbers, not text
- Enables efficient batch processing
- Allows the model to learn character representations

### 3. Data Generator

```python
def data_generator(audio_paths, transcriptions, char_map, batch_size=32):
    while True:
        # Select random batch
        batch_indices = np.random.choice(len(audio_paths), size=batch_size)
        
        # Preprocess audio and transcriptions
        batch_audio = [preprocess_audio(audio_paths[i]) for i in batch_indices]
        batch_transcriptions = [preprocess_transcription(transcriptions[i], char_map) 
                               for i in batch_indices]
        
        # Pad sequences to same length
        padded_audio = tf.keras.preprocessing.sequence.pad_sequences(batch_audio, padding='post')
        padded_transcriptions = tf.keras.preprocessing.sequence.pad_sequences(
            batch_transcriptions, padding='post')
        
        yield inputs, outputs
```

**Why use a generator?**
- **Memory efficiency**: Don't load entire dataset at once
- **Infinite supply**: Can train for multiple epochs
- **Flexibility**: Can apply augmentation or sampling on-the-fly

**Parameter Details:**
- `batch_size=32`: Process 32 samples simultaneously
  - Larger batch: Better gradient estimates, more memory
  - Smaller batch: Less memory, more noisy gradients
  - 32 is a good default

**Why padding?**
- All sequences in a batch must have same shape
- CTC handles padding through length parameters
- `'post'` padding: Add zeros at the end

**Input Dictionary:**
```python
{
    'audio_input': padded_audio,           # Shape: (batch, time_steps, 13)
    'input_length': batch_output_lengths,  # Actual audio lengths (before padding)
    'label_length': batch_input_lengths    # Actual text lengths (before padding)
}
```

CTC uses these lengths to ignore padding positions during loss computation.

### 4. Character Mapping

```python
characters = 'abcdefghijklmnopqrstuvwxyz '
char_map = {char: idx + 1 for idx, char in enumerate(characters)}
char_map['<pad>'] = 0  # Index 0 is reserved for padding
```

**Why reserve index 0?**
- CTC uses index 0 as a "blank" token (special internal token)
- We start character indices from 1
- `char_map['<pad>'] = 0` ensures consistent indexing

**Example:**
```
'a' → 1
'b' → 2
...
'z' → 26
' ' (space) → 27
<pad> → 0
```

### 5. Model Architecture

```python
def build_model(input_shape, output_sequence_length, num_chars):
    input_data = Input(shape=input_shape, name='audio_input')
    
    # LSTM layer: processes sequential audio features
    rnn_layer = LSTM(units=128, return_sequences=True)(input_data)
    
    # TimeDistributed Dense: applies to each time step
    logits = TimeDistributed(Dense(num_chars + 1, activation='softmax'))(rnn_layer)
    
    model = Model(inputs=input_data, outputs=logits)
    model.compile(optimizer=Adam(), loss=CTCLoss(), metrics=['accuracy'])
    return model
```

#### Model Components

**Input Layer:**
- Shape: `(None, 13)` → variable time steps, 13 MFCC features

**LSTM Layer:**
```python
LSTM(units=128, return_sequences=True)
```
- **units=128**: 128 hidden units (neurons in LSTM cell)
  - More units = more capacity but slower training
  - 128 is standard for medium-complexity tasks
  - Too few: underfitting; Too many: overfitting

- **return_sequences=True**: Output all time steps (not just last)
  - Needed because we want character predictions for each frame

**TimeDistributed Dense Layer:**
```python
TimeDistributed(Dense(num_chars + 1, activation='softmax'))
```
- **Why TimeDistributed?**: Applies the same Dense layer to each time step
- **num_chars + 1**: 
  - `num_chars` = 27 (a-z + space)
  - +1 for "blank" token used internally by CTC
  - Total: 28 output classes per time step

- **softmax activation**: Converts outputs to probability distribution
  - Ensures all outputs sum to 1
  - Each output represents P(character | time step)

#### Model Compilation

```python
model.compile(
    optimizer=Adam(),           # Adaptive learning rate optimizer
    loss=CTCLoss(),            # Specialized CTC loss function
    metrics=['accuracy']        # Track accuracy during training
)
```

**Why CTCLoss?**
- Standard cross-entropy assumes 1-to-1 input-output mapping
- CTC handles variable-length sequences and unknown alignments
- Automatically sums over all possible alignments

**Why Adam optimizer?**
- Adaptive learning rate (adjusts per parameter)
- Fast convergence for most tasks
- Better than standard SGD for speech recognition

### 6. Training Parameters

```python
input_shape = (None, 13)        # MFCC shape
output_sequence_length = 200    # Max CTC output length
num_chars = len(char_map)       # 28 (a-z + space + blank)
batch_size = 32                 # Samples per batch
epochs = 10                     # Number of training iterations
```

**Parameter Explanations:**

- **input_shape=(None, 13)**
  - `None`: Variable number of time steps per audio file
  - `13`: Fixed number of MFCC features

- **output_sequence_length=200**
  - Maximum length of text output
  - Longer sequences = more memory
  - Should be longer than expected transcriptions

- **batch_size=32**
  - Trade-off: 16-64 is typical
  - Smaller: More updates, noisier gradients
  - Larger: Fewer updates, smoother gradients

- **epochs=10**
  - One epoch = one pass through entire dataset
  - More epochs = longer training, potential overfitting
  - Monitor validation loss to avoid overfitting

### 7. Training Loop

```python
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=val_generator,
    validation_steps=validation_steps
)
```

**Why use generators?**
- Data doesn't fit in memory all at once
- Continuous data stream during training

**steps_per_epoch calculation:**
```python
steps_per_epoch = len(train_audio_paths) // batch_size
# Example: 1000 samples / 32 batch = ~31 steps per epoch
```

**Validation split:**
- Typically 80% training, 20% validation
- Helps detect overfitting early

### 8. Model Evaluation and Prediction

```python
# Evaluate on test set
test_generator = data_generator(test_audio_paths, test_transcriptions, 
                                char_map, batch_size=1)
test_loss = model.evaluate(test_generator)

# Make prediction
example_audio = preprocess_audio(example_audio_path)
example_audio = np.expand_dims(example_audio, axis=0)  # Add batch dimension
prediction = model.predict(example_audio)

# Decode CTC output to text
decoded_prediction = K.ctc_decode(
    prediction,
    input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
    greedy=True
)[0][0]
```

**Batch dimension:**
- Model expects shape `(batch_size, time_steps, features)`
- Single example needs: `(1, time_steps, 13)`
- `np.expand_dims` adds batch dimension

**CTC Decoding:**
- CTC gives probabilities for each character at each time step
- `ctc_decode`: Converts probabilities → character sequence
- **greedy=True**: Takes highest probability at each step
  - Fast but suboptimal
  - **greedy=False**: Uses beam search (more accurate, slower)

## Why CTC is Better Than Alternatives

| Method | Alignment | Complexity | Performance |
|--------|-----------|-----------|-------------|
| CTC | Automatic | Simple | Excellent |
| HMM-GMM | Manual | Complex | Good |
| Attention | Automatic | Medium | Good |
| CTC + Attention | Automatic | Complex | Excellent |

## Summary of Key Design Choices

| Choice | Parameter | Reason |
|--------|-----------|--------|
| MFCC Features | 13 coefficients | Balance between detail and efficiency |
| Sample Rate | 16 kHz | Standard for speech; captures up to 8 kHz |
| LSTM Units | 128 | Good balance: capacity vs speed |
| Batch Size | 32 | Standard choice; good GPU memory balance |
| Optimizer | Adam | Adaptive learning rate; stable convergence |
| Loss Function | CTCLoss | Handles sequence alignment automatically |
| CTC Decoding | Greedy | Fast inference; beam search for better accuracy |

## How to Use

1. **Prepare data:**
   - Collect audio files (.wav format)
   - Create text transcriptions (lowercase, a-z and space only)
   - Update `audio_paths` and `transcriptions` lists

2. **Train model:**
   ```bash
   python CTC.py
   ```

3. **For inference:**
   - Modify the "Example usage" section
   - Load trained model with `model.load_weights('path/to/weights')`
   - Process new audio and get predictions

## Potential Improvements

1. **Bidirectional LSTM**: Looks at context from both directions
2. **Multiple LSTM layers**: Stacked LSTMs for deeper models
3. **Dropout**: Regularization to prevent overfitting
4. **Data augmentation**: Add noise, time-stretch audio
5. **Beam search decoding**: More accurate than greedy decoding
6. **Language model**: Post-process with n-gram language model

## References

- **CTC Paper**: Graves et al. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks" (2006)
- **MFCC**: Davis & Mermelstein "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences" (1980)
- **LSTM**: Hochreiter & Schmidhuber "Long Short-Term Memory" (1997)

## Dependencies

See `requirements.txt` for all dependencies. Key packages:
- TensorFlow/Keras: Deep learning
- librosa: Audio processing
- NumPy: Numerical computing
- Matplotlib: Visualization
