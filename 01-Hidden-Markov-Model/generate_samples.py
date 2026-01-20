"""
Generate synthetic audio samples for digit recognition training.
Creates sample audio files for digits 0-9 with different frequencies and variations.
"""

import numpy as np
import librosa
import soundfile as sf
import os

def generate_digit_audio(digit, duration=1.0, sr=16000, num_samples=5):
    """
    Generate synthetic audio samples for a digit using sine waves.
    
    Args:
        digit: Digit number (0-9)
        duration: Duration of audio in seconds
        sr: Sample rate
        num_samples: Number of variations to generate
    """
    # Create output directory
    output_dir = './audio_samples'
    digit_dir = os.path.join(output_dir, str(digit))
    os.makedirs(digit_dir, exist_ok=True)
    
    # Base frequency for each digit (200-2000 Hz range)
    base_freq = 200 + (digit * 200)
    
    for i in range(num_samples):
        # Create time array
        t = np.linspace(0, duration, int(sr * duration))
        
        # Generate multiple frequency components for realistic audio
        # Add harmonics and slight variations
        variation = 1.0 + (np.random.rand() - 0.5) * 0.2  # ±10% frequency variation
        
        # Main frequency
        signal = 0.3 * np.sin(2 * np.pi * base_freq * variation * t)
        
        # Add second harmonic
        signal += 0.15 * np.sin(2 * np.pi * base_freq * variation * 2 * t)
        
        # Add third harmonic
        signal += 0.1 * np.sin(2 * np.pi * base_freq * variation * 3 * t)
        
        # Add slight noise for realism
        signal += 0.02 * np.random.randn(len(signal))
        
        # Apply envelope (fade in and out)
        envelope = np.hanning(len(signal))
        signal = signal * envelope
        
        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        
        # Save audio file
        filename = os.path.join(digit_dir, f'digit_{digit}_sample_{i}.wav')
        sf.write(filename, signal, sr)
        print(f"Generated: {filename}")

# Generate samples for all digits
print("Generating synthetic audio samples...")
for digit in range(10):
    generate_digit_audio(digit, num_samples=5)

print("\nAudio sample generation complete!")
print("Samples are saved in: ./audio_samples/")
