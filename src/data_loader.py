"""
Data loading utilities for speech emotion recognition.
"""

from pathlib import Path
import pandas as pd


EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def load_ravdess_data(data_path='data/raw/'):
    """
    Load RAVDESS dataset from the specified path.
    
    RAVDESS filename format: modality-vocal-emotion-intensity-statement-repetition-actor.wav
    """
    data_path = Path(data_path)
    audio_files = []
    
    for wav_file in data_path.rglob('*.wav'):
        filename = wav_file.stem
        parts = filename.split('-')
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion_label = EMOTION_MAP.get(emotion_code)
            
            if emotion_label:
                audio_files.append({
                    'path': str(wav_file),
                    'emotion': emotion_label
                })
    
    df = pd.DataFrame(audio_files)
    return df


if __name__ == '__main__':
    df = load_ravdess_data()
    
    if len(df) == 0:
        print("No audio files found in data/raw/")
        print("Please add RAVDESS dataset files to the data/raw/ directory.")
    else:
        print("Dataset Overview:")
        print(df.head())
        print("\nEmotion Distribution:")
        print(df['emotion'].value_counts())
        print(f"\nTotal samples: {len(df)}")
