"""
Feature extraction utilities for audio processing.
"""

import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40


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


def process_dataset(df, output_dir='data/processed/'):
    """
    Process all audio files in the DataFrame and save features.
    
    Args:
        df: DataFrame with 'path' and 'emotion' columns
        output_dir: Directory to save processed features
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    features = []
    labels = []
    
    print(f"Processing {len(df)} audio files...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        mfcc = extract_mfcc(row['path'])
        
        if mfcc is not None:
            features.append(mfcc)
            labels.append(row['emotion'])
    
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
