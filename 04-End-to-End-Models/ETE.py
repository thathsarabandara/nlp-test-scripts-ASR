import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, TimeDistributed, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CTCLoss
import tensorflow.keras.backend as K


# Load data and preprocess
def preprocess_audio(audio_path):
    """
    Load audio file and extract MFCC features.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        MFCC features transposed (time steps, features)
    """
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    return mfcc.T


def preprocess_transcription(transcription, char_map):
    """
    Convert transcription text to numeric indices using character mapping.
    
    Args:
        transcription: Text transcription
        char_map: Dictionary mapping characters to indices
        
    Returns:
        List of numeric indices
    """
    return [char_map[char] for char in transcription]


def data_generator(audio_paths, transcriptions, char_map, batch_size=32):
    """
    Generator function for feeding batches of data to the model.
    Generates training/validation data in batches for memory efficiency.
    
    Args:
        audio_paths: List of paths to audio files
        transcriptions: List of corresponding transcriptions
        char_map: Dictionary mapping characters to indices
        batch_size: Number of samples per batch (default: 32)
        
    Yields:
        Dictionary with audio inputs, transcriptions, and length information
    """
    while True:
        batch_indices = np.random.choice(len(audio_paths), size=batch_size)
        batch_audio = [preprocess_audio(audio_paths[i]) for i in batch_indices]
        batch_transcriptions = [preprocess_transcription(transcriptions[i], char_map) for i in batch_indices]
        
        batch_input_lengths = np.array([len(seq) for seq in batch_transcriptions])
        batch_output_lengths = np.array([len(seq) for seq in batch_audio])
        
        padded_audio = tf.keras.preprocessing.sequence.pad_sequences(batch_audio, padding='post')
        padded_transcriptions = tf.keras.preprocessing.sequence.pad_sequences(batch_transcriptions, padding='post')
        
        inputs = {
            'audio_input': padded_audio,
            'transcription_input': padded_transcriptions,
            'input_length': batch_output_lengths,
            'label_length': batch_input_lengths
        }
        outputs = {'ctc_output': np.zeros(batch_size)}  # Dummy output for CTC loss
        
        yield inputs, outputs

# Define characters and mapping
characters = 'abcdefghijklmnopqrstuvwxyz '
char_map = {char: idx + 1 for idx, char in enumerate(characters)}  # 0 reserved for blank label
char_map['<pad>'] = 0

# Define the model
def build_model(input_shape, output_sequence_length, num_chars):
    """
    Build an End-to-End CNN+LSTM model for speech recognition.
    Uses CNN for feature extraction from spectrograms, then LSTM for sequence modeling.
    
    Args:
        input_shape: Shape of input audio features (time_steps, features)
        output_sequence_length: Maximum sequence length for output
        num_chars: Number of unique characters (vocabulary size)
        
    Returns:
        Compiled Keras model ready for training
    """
    input_data = Input(shape=input_shape, name='audio_input')
    
    # CNN Feature Extraction Layer
    # Conv2D: Extract spatial-temporal features from MFCC spectrograms
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_data)
    
    # MaxPooling2D: Downsample features, keep most important information
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
    
    # Flatten: Convert 2D feature maps to 1D for LSTM input
    flatten_layer = Flatten()(pool_layer)
    
    # LSTM Layer: Model temporal dependencies in the sequence
    rnn_layer = LSTM(units=128, return_sequences=True)(flatten_layer)
    
    # TimeDistributed Dense: Apply dense layer to each time step
    logits = TimeDistributed(Dense(num_chars + 1, activation='softmax'))(rnn_layer)  # Output includes blank label
    
    model = Model(inputs=input_data, outputs=logits)
    model.compile(optimizer=Adam(), loss=CTCLoss(), metrics=['accuracy'])
    
    return model

# Paths to audio files and corresponding transcriptions
# NOTE: Replace these with your actual data
audio_paths = ['audio_1.wav', 'audio_2.wav']  # List of paths to audio files
transcriptions = ['transcription_1', 'transcription_2']  # List of corresponding transcriptions

# Split data into train and validation sets
split = int(0.8 * len(audio_paths))
train_audio_paths, val_audio_paths = audio_paths[:split], audio_paths[split:]
train_transcriptions, val_transcriptions = transcriptions[:split], transcriptions[split:]

# Parameters
input_shape = (None, 13)  # MFCC feature shape (variable time steps, 13 MFCC coefficients)
output_sequence_length = 200  # Max sequence length for CTC
num_chars = len(char_map)  # Number of unique characters

# Build and compile the model
model = build_model(input_shape, output_sequence_length, num_chars)

# Train the model
batch_size = 32
train_generator = data_generator(train_audio_paths, train_transcriptions, char_map, batch_size=batch_size)
val_generator = data_generator(val_audio_paths, val_transcriptions, char_map, batch_size=batch_size)

steps_per_epoch = len(train_audio_paths) // batch_size
validation_steps = len(val_audio_paths) // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=val_generator,
    validation_steps=validation_steps
)

# Evaluate the model
test_audio_paths, test_transcriptions = val_audio_paths[:10], val_transcriptions[:10]  # Example test set
test_generator = data_generator(test_audio_paths, test_transcriptions, char_map, batch_size=1)
test_loss = model.evaluate(test_generator)
print("Test loss:", test_loss)

# Example usage of the model for prediction
example_audio_path = 'example_audio.wav'
example_audio = preprocess_audio(example_audio_path)
example_audio = np.expand_dims(example_audio, axis=0)
prediction = model.predict(example_audio)
decoded_prediction = K.ctc_decode(
    prediction,
    input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
    greedy=True
)[0][0]
print("Decoded prediction:", decoded_prediction)