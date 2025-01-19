import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
import pandas as pd
import librosa
import numpy as np


class BeehiveDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_dir, snippet_duration=5, num_mfcc=13, sample_rate=16000):
        self.metadata = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.snippet_duration = snippet_duration
        self.num_mfcc = num_mfcc
        self.sample_rate = sample_rate
        self.snippet_samples = self.snippet_duration * self.sample_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_prefix = row["file name"].replace(".raw", "")

        queen_present = 1 if row["queen presence"] == 1 else 0
        label = torch.tensor(queen_present, dtype=torch.float32)

        segment_files = [
            file for file in os.listdir(self.audio_dir)
            if file.startswith(audio_prefix) and file.endswith(".wav")
        ]

        if not segment_files:
            return self.__getitem__((idx + 1) % len(self.metadata))

        segment_files = sorted(segment_files)

        audio_path = os.path.join(self.audio_dir, segment_files[0])

        try:
            waveform, sample_rate = librosa.load(audio_path, sr=self.sample_rate)

            if len(waveform) > self.snippet_samples:
                waveform = waveform[:self.snippet_samples]
            else:
                waveform = np.pad(waveform, (0, self.snippet_samples - len(waveform)))

            mfcc = librosa.feature.mfcc(
                y=waveform, sr=self.sample_rate, n_mfcc=self.num_mfcc
            )

            waveform = torch.tensor(waveform, dtype=torch.float32)
            mfcc = torch.tensor(mfcc, dtype=torch.float32)

            return waveform, mfcc, label

        except Exception as e:
            raise RuntimeError(f"Error processing file {audio_path}: {e}")




if __name__ == '__main__':
    csv_path = "dataset/all_data_updated.csv"
    audio_dir = "dataset/sound_files/sound_files"

    dataset = BeehiveDataset(csv_path, audio_dir)

    waveform, mfcc, label = dataset[0]
    print("Waveform shape:", waveform.shape)
    print("MFCC shape:", mfcc.shape)
    print("Label:", label)
