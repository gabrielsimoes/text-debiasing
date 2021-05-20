import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()

        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.vocab = vocab
        self.ntokens = len(vocab) + 1
        self.d_model = 128
        self.embeddings = Embedding(self.ntokens, self.d_model)
        self.word_mapping = SoftmaxLayer(self.d_model, self.ntokens)

    def forward(self, source):
        return self.word_mapping(self.embeddings(source))

    def decode(self, source):
        output = []
        enc = self.forward(source)
        for b in range(enc.size(0)):
            words = []
            for j in range(enc.size(1)):
                w = self.vocab.denumberize(torch.argmax(enc[b][j]))
                if w == '<EOS>':
                    break
                words.append(w)
            output.append(words)
        return output
