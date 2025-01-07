import torch
import torch.nn as nn

# Example input sequence
batch_size = 2
seq_length = 5
embed_dim = 8
num_heads = 2

input_seq = torch.rand(seq_length, batch_size, embed_dim)  # (S, N, E)

# MultiheadAttention
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

# Linear layers for Q, K, V
query_proj = nn.Linear(embed_dim, embed_dim)
key_proj = nn.Linear(embed_dim, embed_dim)
value_proj = nn.Linear(embed_dim, embed_dim)

# Generate Q, K, V
query = query_proj(input_seq)  # (L, N, E)
key = key_proj(input_seq)      # (S, N, E)
value = value_proj(input_seq)  # (S, N, E)

# MultiheadAttention forward pass
attn_output, attn_output_weights = multihead_attn(query, key, value)

print("Attention Output:", attn_output.shape)
print("Attention Weights:", attn_output_weights.shape)
