"""
Complete demo for HMM-based digit recognition system.
This script generates training data, trains models, and demonstrates recognition.
"""

import numpy as np
import librosa
import os
import sys
from hmmlearn import hmm
import soundfile as sf

def extract_features(audio_file):
    """Extract MFCC features from audio file."""
    try:
        y, sr = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfccs.T
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return None

def generate_synthetic_samples(num_samples_per_digit=5):
    """Generate synthetic audio samples for training."""
    output_dir = './audio_samples'
    
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 10:
        print(f"Audio samples already exist in {output_dir}")
        return
    
    print("Generating synthetic audio samples...")
    sr = 16000
    duration = 1.0
    
    for digit in range(10):
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        
        base_freq = 200 + (digit * 200)
        
        for i in range(num_samples_per_digit):
            t = np.linspace(0, duration, int(sr * duration))
            variation = 1.0 + (np.random.rand() - 0.5) * 0.2
            
            signal = 0.3 * np.sin(2 * np.pi * base_freq * variation * t)
            signal += 0.15 * np.sin(2 * np.pi * base_freq * variation * 2 * t)
            signal += 0.1 * np.sin(2 * np.pi * base_freq * variation * 3 * t)
            signal += 0.02 * np.random.randn(len(signal))
            
            envelope = np.hanning(len(signal))
            signal = signal * envelope
            signal = signal / (np.max(np.abs(signal)) + 1e-8)
            
            filename = os.path.join(digit_dir, f'digit_{digit}_sample_{i}.wav')
            sf.write(filename, signal, sr)
    
    print("✓ Synthetic audio samples generated successfully!")

def train_hmm_models(n_components=3, n_iter=100):
    """Train HMM models for each digit."""
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    data_dir = './audio_samples'
    digit_samples = {}
    
    print("\nExtracting features from training samples...")
    for digit in range(10):
        digit_dir = os.path.join(data_dir, str(digit))
        if not os.path.exists(digit_dir):
            print(f"Warning: No samples found for digit {digit}")
            continue
        
        samples = []
        for filename in os.listdir(digit_dir):
            if filename.endswith('.wav'):
                features = extract_features(os.path.join(digit_dir, filename))
                if features is not None:
                    samples.append(features)
        
        if samples:
            digit_samples[digit] = samples
            print(f"  Digit {digit}: {len(samples)} samples extracted")
    
    print("\nTraining Gaussian HMM models...")
    digit_models = {}
    
    for digit, samples in digit_samples.items():
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
        X = np.concatenate(samples)
        lengths = [sample.shape[0] for sample in samples]
        
        model.fit(X, lengths=lengths)
        digit_models[digit] = model
        print(f"  ✓ Model for digit {digit} trained (converged: {model.monitor_.converged})")
    
    return digit_models

def recognize_digit(audio_file, digit_models):
    """Recognize a digit from audio file using trained models."""
    features = extract_features(audio_file)
    if features is None:
        return None
    
    scores = {}
    for digit, model in digit_models.items():
        try:
            scores[digit] = model.score(features)
        except Exception as e:
            print(f"Error scoring digit {digit}: {e}")
            scores[digit] = -float('inf')
    
    recognized_digit = max(scores, key=scores.get)
    confidence = scores[recognized_digit]
    
    return recognized_digit, confidence, scores

def run_demo():
    """Run the complete demo."""
    print("\n" + "="*60)
    print("HMM-BASED DIGIT RECOGNITION SYSTEM")
    print("="*60)
    
    # Step 1: Generate synthetic samples
    print("\nSTEP 1: Generating/Loading Audio Samples")
    generate_synthetic_samples(num_samples_per_digit=5)
    
    # Step 2: Train models
    print("\nSTEP 2: Training HMM Models")
    digit_models = train_hmm_models(n_components=3, n_iter=100)
    
    if not digit_models:
        print("ERROR: No models were trained!")
        return
    
    # Step 3: Test recognition
    print("\n" + "="*60)
    print("TESTING PHASE - Digit Recognition")
    print("="*60)
    
    test_dir = './audio_samples'
    num_tests = 0
    correct = 0
    
    print("\nTesting on sample data...")
    for digit in range(10):
        digit_dir = os.path.join(test_dir, str(digit))
        if not os.path.exists(digit_dir):
            continue
        
        for filename in os.listdir(digit_dir)[:2]:  # Test on first 2 samples
            if filename.endswith('.wav'):
                filepath = os.path.join(digit_dir, filename)
                result = recognize_digit(filepath, digit_models)
                
                if result:
                    recognized_digit, confidence, scores = result
                    is_correct = recognized_digit == digit
                    if is_correct:
                        correct += 1
                    num_tests += 1
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} File: {filename}")
                    print(f"   Actual: {digit}, Recognized: {recognized_digit}, Confidence: {confidence:.4f}")
    
    # Step 4: Results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if num_tests > 0:
        accuracy = (correct / num_tests) * 100
        print(f"Tests conducted: {num_tests}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
    
    # Step 5: Interactive test
    print("\n" + "="*60)
    print("INTERACTIVE TEST")
    print("="*60)
    
    print("\nGenerating a test audio for digit 5...")
    test_audio = './test_audio_digit5.wav'
    
    # Create a synthetic test sample for digit 5
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    base_freq = 200 + (5 * 200)
    
    signal = 0.3 * np.sin(2 * np.pi * base_freq * 1.05 * t)
    signal += 0.15 * np.sin(2 * np.pi * base_freq * 1.05 * 2 * t)
    signal += 0.1 * np.sin(2 * np.pi * base_freq * 1.05 * 3 * t)
    signal += 0.02 * np.random.randn(len(signal))
    envelope = np.hanning(len(signal))
    signal = signal * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    sf.write(test_audio, signal, sr)
    
    result = recognize_digit(test_audio, digit_models)
    if result:
        recognized_digit, confidence, scores = result
        print(f"Test audio: {test_audio}")
        print(f"Actual digit: 5")
        print(f"Recognized digit: {recognized_digit}")
        print(f"Confidence: {confidence:.4f}")
        print(f"\nTop 3 predictions:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (digit, score) in enumerate(sorted_scores[:3], 1):
            print(f"  {rank}. Digit {digit}: {score:.4f}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_demo()
