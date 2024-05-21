import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class EventEmbedding(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super(EventEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):

    def __init__(self, input_dim, embed_dim, nhead, num_encoder_layers, dim_feedforward, max_event_count):
        super(TransformerModel, self).__init__()
        self.__embedding = EventEmbedding(input_dim=input_dim, embed_dim=embed_dim)
        self.__positional_encoding = PositionalEncoding(embed_dim, max_len=max_event_count)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.__transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.__fc_out = nn.Linear(embed_dim, 1)
        self.__criterion = nn.BCEWithLogitsLoss()

    def to(self, device):
        self.__device = device
        return super().to(device)

    def __padding_mask(self, mask):
        return (1 - mask).bool()
        
    def forward(self, x, mask):
        batch_size, max_event_count, _ = x.size()
        
        x = x.view(-1, x.size(-1))
        x = self.__embedding(x)
        x = x.view(batch_size, max_event_count, -1)
        x = self.__positional_encoding(x)

        causal_mask = torch.triu(torch.ones(max_event_count, max_event_count), diagonal=1).bool().to(self.__device)
        
        x = self.__transformer_encoder(x, mask=causal_mask, src_key_padding_mask=self.__padding_mask(mask), is_causal=True)

        output = self.__fc_out(x)
        output = torch.cumsum(output.squeeze(), dim=1)
        output = torch.sigmoid(output)

        return output

    def run_training_step(self, X, y, mask):
        return self.__criterion(self(X, mask), y)

    def accuracy(self, X, y, mask):
        with torch.no_grad():
            return (self(X, mask).round().squeeze().eq(y) * mask).sum() / mask.sum()
