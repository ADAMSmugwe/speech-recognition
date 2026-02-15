56543456"""
Feature extraction utilities for audio processing.
"""

import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40


def add_noise(data, noise_factor=0.005):
    """Add random Gaussian noise to audio data."""
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def shift(data, shift_max=0.2):
    """Randomly shift audio data."""
    shift_amount = int(np.random.uniform(-shift_max, shift_max) * len(data))
    return np.roll(data, shift_amount)


def stretch(data, rate=0.8):
    """Time stretch audio data."""
    return librosa.effects.time_stretch(data, rate=rate)


def pitch(data, sr, n_steps=2):
    """Pitch shift audio data."""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)


def extract_mfcc(file_path, sr=SAMPLE_RATE, duration=DURATION, n_mfcc=N_MFCC):
    """
    Extract MFCC features from an audio file.
    
    Args:
        file_path: Path to the audio file
        sr: Target sample rate (default: 22050Hz)
        duration: Fixed duration in seconds (default: 3s)
        n_mfcc: Number of MFCC coefficients (default: 40)
    
    Returns:
        MFCC features as numpy array
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        max_len = sr * duration
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_mfcc_from_audio(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract MFCC features from raw audio data."""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None


def process_dataset(df, output_dir='data/processed/', augment=True):
    """
    Process all audio files in the DataFrame and save features.
    
    Args:
        df: DataFrame with 'path' and 'emotion' columns
        output_dir: Directory to save processed features
        augment: Whether to include augmented versions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    features = []
    labels = []
    
    total_samples = len(df) * (2 if augment else 1)
    print(f"Processing {len(df)} audio files (with augmentation: {total_samples} total samples)...")
    
    with tqdm(total=total_samples, desc="Extracting MFCCs") as pbar:
        for _, row in df.iterrows():
            try:
                audio, _ = librosa.load(row['path'], sr=SAMPLE_RATE, duration=DURATION)
                
                max_len = SAMPLE_RATE * DURATION
                if len(audio) < max_len:
                    audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
                else:
                    audio = audio[:max_len]
                
                mfcc_original = extract_mfcc_from_audio(audio)
                if mfcc_original is not None:
                    features.append(mfcc_original)
                    labels.append(row['emotion'])
                    pbar.update(1)
                
                if augment:
                    audio_noisy = add_noise(audio)
                    mfcc_augmented = extract_mfcc_from_audio(audio_noisy)
                    if mfcc_augmented is not None:
                        features.append(mfcc_augmented)
                        labels.append(row['emotion'])
                        pbar.update(1)
                        
            except Exception as e:
                print(f"\nError processing {row['path']}: {e}")
                continue
    
    features = np.array(features)
    labels = np.array(labels)
    
    np.save(output_path / 'features.npy', features)
    np.save(output_path / 'labels.npy', labels)
    
    print(f"\nSaved {len(features)} features to {output_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return features, labels


if __name__ == '__main__':
    from data_loader import load_ravdess_data
    
    df = load_ravdess_data()
    
    if len(df) == 0:
        print("No audio files found. Please add RAVDESS dataset to data/raw/")
    else:
        features, labels = process_dataset(df)
        print("\nFeature extraction complete!")
