import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Function to extract MFCC features from audio
def extract_features(audio_path):
    """
    Extract MFCC features from an audio file.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        MFCC feature array (n_mfcc, time_steps)
    """
    audio, sr = librosa.load(audio_path, sr=22050)  # Standard sample rate for audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs


# Load and preprocess audio data
# NOTE: Replace these with your actual data paths and labels
audio_paths = ['audio_1.wav', 'audio_2.wav']  # List of paths to audio files
labels = ['digit_0', 'digit_1']  # Corresponding labels (for speech digit recognition)

# Encode labels to numeric values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)


# Function to prepare audio for transfer learning
def prepare_audio_for_vgg(mfcc_features, target_size=(224, 224)):
    """
    Convert MFCC features to 224x224x3 format for VGG16.
    
    Strategy: Convert 2D MFCC (13, time_steps) to 3-channel 224x224 image
    by: 1) Time-stretching/padding to 224 time steps
        2) Stacking MFCC coefficients to create depth
        3) Replicating to get 3 channels (RGB)
    
    Args:
        mfcc_features: MFCC array of shape (13, time_steps)
        target_size: Target size for VGG16 (224, 224)
        
    Returns:
        Array of shape (224, 224, 3) ready for VGG16
    """
    n_mfcc, n_frames = mfcc_features.shape
    
    # Pad or truncate time steps to match target width (224)
    if n_frames < target_size[1]:
        # Pad with zeros
        padded = np.pad(mfcc_features, 
                       ((0, 0), (0, target_size[1] - n_frames)), 
                       mode='constant', constant_values=0)
    else:
        # Truncate
        padded = mfcc_features[:, :target_size[1]]
    
    # Replicate MFCC coefficients to fill 224x224 height
    # Strategy: Interpolate 13 coefficients to 224 pixels (using repeat)
    audio_image = np.repeat(padded, target_size[0] // n_mfcc, axis=0)[:target_size[0], :]
    
    # Convert to 3-channel (RGB) by replicating across channels
    # This creates a grayscale-like representation
    rgb_image = np.stack([audio_image, audio_image, audio_image], axis=2)
    
    # Normalize to [0, 1] range for VGG16
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    
    return rgb_image


# Build transfer learning model
def build_transfer_learning_model(num_classes, freeze_base=True):
    """
    Build a transfer learning model using VGG16 as feature extractor.
    
    Args:
        num_classes: Number of output classes
        freeze_base: Whether to freeze VGG16 weights (True = feature extraction mode)
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained VGG16 (trained on ImageNet)
    # weights='imagenet': Use pre-trained weights from ImageNet
    # include_top=False: Exclude classification layers, keep only feature extraction part
    # input_shape=(224, 224, 3): Standard VGG16 input size
    base_model = VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(224, 224, 3))
    
    # Freeze pre-trained layers to keep learned features
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Build custom classification head
    model = Sequential([
        base_model,
        # Flatten: Convert (7, 7, 512) feature maps to 1D vector
        # VGG16 outputs 7x7x512 = 25,088 features
        Flatten(),
        
        # Dense layer for feature combination
        Dense(256, activation='relu'),  # 256 units to combine VGG16 features
        
        # Output layer
        Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Lower learning rate for fine-tuning
        loss='sparse_categorical_crossentropy',  # For integer labels
        metrics=['accuracy']
    )
    
    return model


# Prepare dataset
print("Loading and preprocessing audio data...")

# Extract features from all audio files
all_features = []
for audio_path in audio_paths:
    mfcc = extract_features(audio_path)
    # Convert MFCC to VGG16-compatible format
    vgg_compatible = prepare_audio_for_vgg(mfcc)
    all_features.append(vgg_compatible)

X = np.array(all_features)
y = encoded_labels

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {num_classes}")

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42,  # Reproducibility
    stratify=y  # Maintain class distribution
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Build the transfer learning model
print("Building transfer learning model...")
model = build_transfer_learning_model(num_classes, freeze_base=True)

print(model.summary())

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

# Evaluate the model on test set
print("Evaluating the model on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Make predictions on new audio
print("Making predictions on test samples...")
predictions = model.predict(X_test[:5])  # Predict on first 5 test samples
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

print("Sample predictions:")
for i, (pred_label, true_label) in enumerate(zip(predicted_labels, 
                                                   label_encoder.inverse_transform(y_test[:5]))):
    print(f"  Sample {i+1}: Predicted = {pred_label}, Actual = {true_label}")