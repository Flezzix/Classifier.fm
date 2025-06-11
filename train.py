import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import librosa
import numpy as np
import soundfile as sf

# Parameters
DATA_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 22050
N_MELS = 128

# Genres: folder names as labels
GENRES = sorted(os.listdir(DATA_DIR))
genre_to_idx = {g: i for i, g in enumerate(GENRES)}

def load_audio_sf(file_path, sr=SAMPLE_RATE):
    y, original_sr = sf.read(file_path)
    if original_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
    return y, sr

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, root_dir, genres, transform=None):
        self.samples = []
        self.genres = genres
        self.transform = transform

        for genre in genres:
            genre_dir = os.path.join(root_dir, genre)
            for filename in os.listdir(genre_dir):
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    self.samples.append((os.path.join(genre_dir, filename), genre_to_idx[genre]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        # Load audio
        y, sr = load_audio_sf(file_path, sr=SAMPLE_RATE)
        # Convert to Mel spectrogram
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        # Normalize to 0-1
        melspec_norm = (melspec_db + 80) / 80  # typical dB range normalization
        # Convert to 3 channels by duplicating (to match ResNet input)
        img = np.stack([melspec_norm]*3, axis=0).astype(np.float32)

        if self.transform:
            img = self.transform(torch.tensor(img))

        return img, label

# DataLoader and transforms
# ResNet18 expects (224, 224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

dataset = AudioDataset(DATA_DIR, GENRES, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load pretrained ResNet18 and modify final layer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")

# Save the fine-tuned model
torch.save(model.state_dict(), "model/resnet_genre.pth")
print("Training complete and model saved!")
