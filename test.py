import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

# ✅ Ensure MP3 support
torchaudio.set_audio_backend("sox_io")


# ---------------------------
# Dataset
# ---------------------------
class DatasetsCustom(Dataset):
    def __init__(self, X, y, path, sample_rate=16000, n_mfcc=40):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.path = path
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

    def load_audio(self, file_name):
        audio_path = os.path.join(self.path, file_name)
        waveform, sr = torchaudio.load(audio_path)

        # ✅ Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # shape: (time, n_mfcc)
        return mfcc

    def __getitem__(self, idx):
        file_name = self.X.iloc[idx]
        label = self.y.iloc[idx]
        mfcc = self.load_audio(file_name)
        return mfcc, label

    def __len__(self):
        return len(self.X)


# ---------------------------
# Model (RNN with LSTM)
# ---------------------------
class Model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_shape,
            hidden_size=hidden_units,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_units * 2, output_shape)  # *2 for bidirectional

    def forward(self, x: torch.Tensor):
        # x: (batch, time, features)
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]  # take last timestep
        output = self.fc(lstm_out_last)
        return output


# ---------------------------
# Collate Function (pad sequences)
# ---------------------------
def collate_fn(batch):
    # batch = [(mfcc, label), ...]
    features = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    # Pad to same length
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    return features_padded, labels


# ---------------------------
# Training Script
# ---------------------------
if __name__ == "__main__":
    # Example (replace with your actual train/test split)
    import pandas as pd

    # Example: assume train/test CSV files contain filenames and labels
    train_csv = pd.read_csv("train.csv")  # columns: ["filename", "label"]
    test_csv = pd.read_csv("test.csv")

    image_path = "data/train"  # folder with MP3 files

    train_dataset = DatasetsCustom(
        X=train_csv["filename"],
        y=train_csv["label"],
        path=image_path
    )

    test_dataset = DatasetsCustom(
        X=test_csv["filename"],
        y=test_csv["label"],
        path=image_path
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Model
    input_shape = 40   # n_mfcc
    hidden_units = 128
    output_shape = len(train_csv["label"].unique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(input_shape, hidden_units, output_shape).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")


# ------------------ SAVE MODEL ------------------
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH / "model_train.pth")
print("✅ Model saved at models/model_train.pth")
