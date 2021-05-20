import math
import torch

from layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()

        self.vocab = vocab
        self.ntokens = len(vocab) + 1
        # self.d_model = 128
        # self.d_ff = 64
        # self.n_heads = 2
        # self.n_layers = 2
        self.d_model = 1024
        self.d_ff = 512
        self.n_heads = 4
        self.n_layers = 4
        self.dropout = 0.1
        self.copy = True
        self.coverage = True

        self.emb_src = Embeddings(self.ntokens, self.d_model)
        self.emb_tgt = Embeddings(self.ntokens, self.d_model)
        self.pos_enc_src = PositionalEncoding(self.d_model, self.dropout)
        self.pos_enc_tgt = PositionalEncoding(self.d_model, self.dropout)
        self.encoder = TransformerEncoder(self.d_model, self.d_ff, self.n_heads, self.n_layers, self.dropout)
        self.decoder = TransformerDecoder(self.d_model, self.d_ff, self.n_heads, self.n_layers, self.dropout, self.coverage)
        self.softmax = SoftmaxLayer(self.d_model, self.ntokens)

        self.bos_token = self.vocab.numberize('<BOS>')
        self.eos_token = self.vocab.numberize('<EOS>')
        self.cpy_token = self.vocab.numberize('<CPY>')

    def forward(self, src, tgt):
        src_mask = (src != 0).unsqueeze(-2)
        tgt_mask = (tgt != 0).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.autograd.Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        enc = self._encode(src, src_mask)
        dec, alpha, attns, coverages = self._decode(enc, tgt, src_mask, tgt_mask)
        return self._predict(dec, alpha, src), attns, coverages

    def _encode(self, src, src_mask):
        src = self.pos_enc_src(self.emb_src(src))
        enc = self.encoder(src, src_mask)
        return enc

    def _decode(self, enc, tgt, src_mask, tgt_mask):
        tgt = self.pos_enc_tgt(self.emb_tgt(tgt))
        dec, attns, coverages = self.decoder(enc, tgt, src_mask, tgt_mask)
        alpha = torch.mean(attns[-1], dim=-3)
        return dec, alpha, attns, coverages

    def _predict(self, dec, alpha=None, src=None):
        if self.copy and alpha is not None:
            output = self.softmax(dec, log=False)
            cpy_prob = output[:, :, self.cpy_token]
            alpha = alpha * cpy_prob.unsqueeze(-1)
            src_expanded = src.unsqueeze(-2).repeat(1, output.size(1), 1)
            output = output.scatter_add(-1, src_expanded, alpha)
            output = torch.log(output + 1e-10)
        else:
            output = self.softmax(dec, log=True)
        return output

    def decode(self, src):
        max_len = 200
        src_mask = (src != 0).unsqueeze(-2)

        enc = self._encode(src, src_mask)
        ys = torch.ones(src.size(0), 1).fill_(self.bos_token).type_as(src.data)
        for i in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(-1)).to(device)
            out, alpha, _attns, _coverage = self._decode(enc, ys, src_mask, tgt_mask)
            prob = self._predict(out[:, -1].unsqueeze(1), alpha[:, -1].unsqueeze(1), src).squeeze(1)
            if self.copy:
                prob[:, self.cpy_token] = -np.inf
            next_word = torch.argmax(prob, dim=-1)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        results = []
        for words in ys:
            results.append([])
            for w in words[1:]:
                if w == self.eos_token:
                    break
                results[-1].append(self.vocab.denumberize(w))
        return results
