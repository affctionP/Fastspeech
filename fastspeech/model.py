import torch
import torch.nn as nn
import torch.nn.functional as F
from phonemizing import phoneme_vocab
import torch.optim as optim
from DataSet import LJSpeechDataset
from torch.utils.data import DataLoader
# FFT Block (Feed-Forward Transformer Block)
class FFTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, conv_kernel_size, conv_expansion):
        super(FFTBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, conv_expansion, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)
        self.conv2 = nn.Conv1d(conv_expansion, hidden_size, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        
        # 1D Convolution with residual connection
        x_conv = self.conv1(x.transpose(1, 2))
        x_conv = self.relu(x_conv)
        x_conv = self.conv2(x_conv).transpose(1, 2)
        x = x + self.dropout(x_conv)
        x = self.layer_norm2(x)
        return x

# Duration Predictor
class DurationPredictor(nn.Module):
    def __init__(self, hidden_size, conv_kernel_size):
        super(DurationPredictor, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x.transpose(1, 2))
        x = self.relu(x).transpose(1, 2)
        x = self.layer_norm1(x)
        x = self.conv2(x.transpose(1, 2))
        x = self.relu(x).transpose(1, 2)
        x = self.layer_norm2(x)
        x = self.linear(x)
        return x.squeeze(-1)

# FastSpeech Model
class FastSpeech(nn.Module):
    def __init__(self, vocab_size=51, hidden_size=384, num_heads=2, num_blocks=6, conv_kernel_size=3, conv_expansion=1536, mel_dim=80):
        super(FastSpeech, self).__init__()
        # Phoneme embedding
        self.phoneme_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # FFT blocks for encoder and decoder
        self.encoder = nn.ModuleList([FFTBlock(hidden_size, num_heads, conv_kernel_size, conv_expansion) for _ in range(num_blocks)])
        self.decoder = nn.ModuleList([FFTBlock(hidden_size, num_heads, conv_kernel_size, conv_expansion) for _ in range(num_blocks)])
        
        # Length regulator
        self.duration_predictor = DurationPredictor(hidden_size, conv_kernel_size)
        
        # Output linear layer
        self.linear = nn.Linear(hidden_size, mel_dim)

    def forward(self, phonemes, durations=None, mel_targets=None):
        # Embed phonemes
        x = self.phoneme_embedding(phonemes)
        
        # Encoder
        for block in self.encoder:
            x = block(x)
        
        # Predict durations
        predicted_durations = self.duration_predictor(x)
        
        # Use durations for length regulation
        if durations is not None:
            x = self.length_regulator(x, durations)
        
        # Decoder
        for block in self.decoder:
            x = block(x)
        
        # Convert to mel-spectrogram
        mel_output = self.linear(x)
        
        return mel_output, predicted_durations

    @staticmethod
    def length_regulator(x, durations):
        # Upsample phoneme representations according to durations
        expanded = []
        for i, duration in enumerate(durations):
            expanded.append(x[i].repeat_interleave(duration, dim=0))
        return nn.utils.rnn.pad_sequence(expanded, batch_first=True)

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 18
    num_epochs = 100
    learning_rate = 1e-4
    metadata_path ="/home/hosseini/Downloads/LJSpeech-1.1/metadata.csv"
    wave_dir="/home/hosseini/Downloads/LJSpeech-1.1/wavs"

    # Prepare dataset and DataLoader
    dataset = LJSpeechDataset(metadata_path=metadata_path, wav_dir=wave_dir, phoneme_vocab=phoneme_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))

    # Model, loss functions, and optimizer
    model = FastSpeech(vocab_size=len(phoneme_vocab))
    criterion_mel = nn.MSELoss()
    criterion_duration = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_mel_loss = 0
        epoch_duration_loss = 0

        for batch in dataloader:
            phonemes, mels = batch
            for i, mel in enumerate(mels):
                print(f"Tensor {i}: shape {mel.shape}")

            phonemes = torch.nn.utils.rnn.pad_sequence(phonemes, batch_first=True)
            mels = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True)

            # Forward pass
            predicted_mels, predicted_durations = model(phonemes)
            
            # Compute target durations (lengths of each mel sequence)
            target_durations = torch.tensor([len(m) for m in mels], dtype=torch.float32)

            # Compute losses
            loss_mel = criterion_mel(predicted_mels, mels)
            loss_duration = criterion_duration(predicted_durations, target_durations)

            # Backward pass
            optimizer.zero_grad()
            loss = loss_mel + loss_duration
            loss.backward()
            optimizer.step()

            epoch_mel_loss += loss_mel.item()
            epoch_duration_loss += loss_duration.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Mel Loss: {epoch_mel_loss:.4f}, Duration Loss: {epoch_duration_loss:.4f}")

