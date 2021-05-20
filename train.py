import collections.abc
import math
import random
import torch
import time
import numpy as np

from baseline import BaselineModel
from transformer import TransformerModel

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch_size = 1
batch_size = 20
max_len = 200
# max_len = 50
pad_id = 0
coverage_lambda = 0.5

def read_data(filename):
    source = []
    target = []
    for line in open(filename):
        _0, source_line, target_line, _1, _2, _3, _4 = line.split('\t')
        source_words = ['<BOS>'] + source_line.split() + ['<EOS>']
        target_words = ['<BOS>'] + target_line.split() + ['<EOS>']

        if len(source_words) > max_len or len(target_words) > max_len:
            continue

        source.append(source_words)
        target.append(target_words)

    return source, target

def index_data(data, vocab):
    return [
        torch.tensor([vocab.numberize(w) for w in sentence] + [pad_id] * (max_len - len(sentence)), dtype=torch.long, device=device)
        for sentence in data
    ]

def batchify_data(data):
    nbatch = len(data) // batch_size
    return torch.cat(data[:nbatch * batch_size]).view(nbatch, batch_size, max_len).to(device)


class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<CPY>']
        self.num_to_word = list(words)
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(elf, word):
        raise NotImplementedError()
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else:
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num] if num < len(self.num_to_word) else '<UNK>'


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--dev', type=str, help='development data')
    parser.add_argument('--test', type=str, help='test data')
    parser.add_argument('infile', nargs='?', type=str, help='test data to translate')
    parser.add_argument('-o', '--outfile', type=str, help='write translations to file')
    parser.add_argument('--load', type=str, help='load model from file')
    parser.add_argument('--save', type=str, help='save model in file')
    parser.add_argument('--model', type=str, help='model to train')
    args = parser.parse_args()

    if args.train:
        # Read training data and create vocabularies
        train_source, train_target = read_data(args.train)

        vocab = Vocab()
        for words in train_source:
            vocab |= words
        for words in train_target:
            vocab |= words

        train_source = index_data(train_source, vocab)
        train_target = index_data(train_target, vocab)

        # Read dev data
        if args.dev is None:
            print('error: --dev is required', file=sys.stderr)
            sys.exit()
        dev_source, dev_target = read_data(args.dev)
        dev_source = index_data(dev_source, vocab)
        dev_target = index_data(dev_target, vocab)

    if args.load:
        model = torch.load(args.load, map_location=device)

        vocab = model.vocab

        test_source, _test_target = read_data(args.test)
        test_source = index_data(test_source, vocab)
        for i, source in enumerate(batchify_data(test_source)):
            output = model.decode(source)
            for words in output:
                print(' '.join(words))
        exit(0)

    if args.model == 'baseline':
        model = BaselineModel(vocab).to(device)
    elif args.model == 'transformer':
        model = TransformerModel(vocab).to(device)
    else:
        print('error: invalid model or model not specified (--model)', file=sys.stderr)
        sys.exit()

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    lr = 5 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    bos_token = vocab.numberize('<BOS>')
    eos_token = vocab.numberize('<EOS>')
    cpy_token = vocab.numberize('<CPY>')

    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()

        trainbatches = list(zip(batchify_data(train_source), batchify_data(train_target)))
        random.shuffle(trainbatches)

        for i, (source, target) in enumerate(trainbatches):
            optimizer.zero_grad()
            if args.model == 'baseline':
                output = model(source)
                loss = criterion(output.reshape(-1, len(model.vocab) + 1), target.reshape(-1))
            else:
                target_no_bos = target[target!=bos_token].view(target.size(0), -1)
                target_no_eos = target[target!=eos_token].view(target.size(0), -1)
                output, attns, coverages = model(source, target_no_eos)
                loss = criterion(output.reshape(-1, len(model.vocab) + 1), target_no_bos.reshape(-1))
                for attn, coverage in zip(attns, coverages):
                    loss += coverage_lambda * torch.sum(torch.min(attn, coverage)) / (attn.size(0) * attn.size(1) * attn.size(2))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 20
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, len(train_source) // batch_size, scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate():
        print('Evaluating...')
        model.eval() # Turn on the evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for i, (source, target) in enumerate(zip(batchify_data(dev_source), batchify_data(dev_target))):
                if args.model == 'baseline':
                    output = model(source)
                    loss = criterion(output.reshape(-1, len(model.vocab) + 1), target.reshape(-1))
                else:
                    target_no_bos = target[target!=bos_token].view(target.size(0), -1)
                    target_no_eos = target[target!=eos_token].view(target.size(0), -1)
                    output, attns, coverages = model(source, target_no_eos)
                    loss = criterion(output.reshape(-1, len(model.vocab) + 1), target_no_bos.reshape(-1))
                    for attn, coverage in zip(attns, coverages):
                        loss += coverage_lambda * torch.sum(torch.min(attn, coverage)) / (attn.size(0) * attn.size(1) * attn.size(2))

                total_loss += batch_size * loss
            for i, source in enumerate(batchify_data(dev_source)[:1]):
                output = model.decode(source[:10])
                for words in output:
                    print(' '.join(words))
        return total_loss / len(dev_source)

    best_val_loss = float("inf")
    epochs = 100 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        print('Epoch', epoch)
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = epoch

        print('Saving', epoch, 'Best', best_model)
        torch.save(model, args.model+"-"+str(epoch)+".model")

        scheduler.step()
