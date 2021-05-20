import torch
import numpy as np
import math
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# From "The Annotated Transformer"
class Embeddings(torch.nn.Module):
    def __init__(self, vocab_size, output_dims):
        super(Embeddings, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, output_dims)

    def forward(self, inp):
        emb = self.embedding(inp)
        return emb * math.sqrt(emb.size(-1))


# From "The Annotated Transformer"
class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# Provided for class, modified
class SoftmaxLayer(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SoftmaxLayer, self).__init__()
        self.proj = torch.nn.Linear(input_dims, output_dims)

    def forward(self, inp, log=True):
        if log:
            return torch.log_softmax(self.proj(inp), dim=-1)
        else:
            return torch.softmax(self.proj(inp), dim=-1)


# From "The Annotated Transformer"
def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# From "The Annotated Transformer"
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# From "The Annotated Transformer"
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


# From "The Annotated Transformer"
class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Provided for class, modified based on "The Annotated Transformer" for multiple heads
class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.proj_q = torch.nn.Linear(d_model, d_model)
        self.proj_k = torch.nn.Linear(d_model, d_model)
        self.proj_v = torch.nn.Linear(d_model, d_model)
        self.proj_out = torch.nn.Linear(d_model, d_model, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask, do_proj=True):
        # linear projection on q, k, v, split into heads
        q, k, v = [proj(x).view(x.size(0), -1, self.n_heads, self.d_head).transpose(1, 2)
                   for x, proj in zip([q, k, v], [self.proj_q, self.proj_k, self.proj_v])]

        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        # compute self attention
        logits = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        if mask is not None:
            # same mask for multiple heads
            mask = mask.unsqueeze(1)
            logits = logits.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        saved_weights = weights
        weights = self.dropout(weights)
        context = weights @ v

        # concat different heads
        context = context.transpose(1, 2).contiguous().view(q.size(0), -1, self.d_model)

        # final projection
        return saved_weights, self.proj_out(context)


# Implemented from scratch
class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.attns = clones(MultiHeadedAttention(d_model, n_heads, dropout), n_layers)
        self.ffs = clones(FeedForward(d_model, d_ff, dropout), n_layers)
        self.norms_1 = clones(LayerNorm(d_model), n_layers)
        self.norms_2 = clones(LayerNorm(d_model), n_layers)
        self.final_norm = LayerNorm(d_model)
        self.dropouts_1 = clones(torch.nn.Dropout(dropout), n_layers)
        self.dropouts_2 = clones(torch.nn.Dropout(dropout), n_layers)

    def forward(self, x, src_mask):
        for i in range(self.n_layers):
            save = x
            x = self.norms_1[i](x)
            weights, x = self.attns[i](x, x, x, src_mask)
            x = self.dropouts_1[i](x)
            x = x + save

            save = x
            x = self.norms_2[i](x)
            x = self.ffs[i](x)
            x = self.dropouts_2[i](x)
            x = x + save
        x = self.final_norm(x)
        return x


# Implemented from scratch
class TransformerDecoder(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, coverage):
        super(TransformerDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.self_attns = clones(MultiHeadedAttention(d_model, n_heads, dropout), n_layers)
        self.src_attns = clones(MultiHeadedAttention(d_model, n_heads, dropout), n_layers)
        self.ffs = clones(FeedForward(d_model, d_ff, dropout), n_layers)
        self.norms_1 = clones(LayerNorm(d_model), n_layers)
        self.norms_2 = clones(LayerNorm(d_model), n_layers)
        self.norms_3 = clones(LayerNorm(d_model), n_layers)
        self.final_norm = LayerNorm(d_model)
        self.dropouts_1 = clones(torch.nn.Dropout(dropout), n_layers)
        self.dropouts_2 = clones(torch.nn.Dropout(dropout), n_layers)
        self.dropouts_3 = clones(torch.nn.Dropout(dropout), n_layers)
        self.coverage = coverage

    def forward(self, enc, x, src_mask, tgt_mask):
        attns = []
        coverages = []
        coverage_sum = torch.zeros(x.size(0), self.n_heads, x.size(1), enc.size(1)).to(device)

        for i in range(self.n_layers):
            save = x
            x = self.norms_1[i](x)
            attn, x = self.self_attns[i](x, x, x, tgt_mask)
            x = self.dropouts_1[i](x)
            x = x + save

            save = x
            x = self.norms_2[i](x)
            attn, x = self.src_attns[i](x, enc, enc, src_mask)
            x = self.dropouts_2[i](x)
            x = x + save

            attns.append(attn)
            if self.coverage:
                coverage_sum += attn
                coverages.append(coverage_sum.clone())

            save = x
            x = self.norms_3[i](x)
            x = self.ffs[i](x)
            x = self.dropouts_3[i](x)
            x = x + save
        x = self.final_norm(x)
        return x, attns, coverages
