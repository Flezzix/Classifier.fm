import torchaudio
import torch

SAMPLE_RATE = 22050

def preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=128
    )(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    mel_spec = mel_spec[:, :128, :128]
    return mel_spec
