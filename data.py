import torch
import os
import random
import math
import time
from torch.autograd import Variable
from collections import OrderedDict, Counter
from itertools import islice
import const
from random import shuffle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############### Load text by tokens as in MOS ###########################
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = []
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
            ids = torch.tensor(ids, device=device, dtype=torch.long)
        return ids
    
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    return data

def get_batch(source, i, seq_len=35):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len, :].t().contiguous()
    target = source[i+1:i+1+seq_len, :].t().contiguous()
    return data, target

###############################################

class DataSet:
    def __init__(self, datapath, batch_size=1, build_dict=False, display_freq=0, max_dict=10000, max_len=100, trunc_len=100):
        
        self.dictionary = {}
        self.id2w = []
        self.frequency = Counter()
        self.sentence = []
        
        self.batch_size = batch_size
        self.datapath = datapath
        self.num_batch = 0
        self.num_tokens = 0
        self.num_vocb = 0
        self.shuffle_level = 2
        self.display_freq = display_freq
        self.max_dict = max_dict
        self.max_len = max_len
        
        assert trunc_len <= max_len, 'trunc length should be smaller than maximum lenth'
        self.trunc_len = trunc_len
        print('='*89)
        print('Loading data from %s ...' % datapath)
        
        if build_dict:
            self.build_dict()


    def describe_dataset(self):
        print('='*89)
        print('Data discription:')
        print('Data name : %s' % self.datapath)
        print('Number of sentence : %d' % len(self.sentence))
        print('Number of tokens : %d' % self.num_tokens)
        print('Vocabulary size : %d' % self.num_vocb)
        print('Number of batches : %d' % self.num_batch)
        print('Batch size : %d' % self.batch_size)


    def build_dict(self, save_as_text=True):
        
        print('Building dictionary...')
        
        with open(self.datapath, 'r') as f:
            self.num_tokens = 0
            self.num_vocb = 0
            
            for count, line in enumerate(f):
                
                if self.display_freq > 0 and count % self.display_freq == 0:
                    print('%d sentence processed'%(count))

                tokens = [const.BOS_WORD] + line.split() + [const.EOS_WORD]
                
                self.frequency.update(tokens)

            max_freq = max(self.frequency.values()) 
            self.frequency[const.UNK_WORD] = 4 - const.UNK + max_freq
            self.frequency[const.BOS_WORD] = 4 - const.BOS + max_freq
            self.frequency[const.EOS_WORD] = 4 - const.EOS + max_freq 
            self.frequency[const.PAD_WORD] = 4 - const.PAD + max_freq
            
            self.frequency = sorted(self.frequency.items(), key=lambda tup: tup[0])
            self.frequency.sort(key=lambda tup: tup[1], reverse=True)

            for token, freq in self.frequency:
                if self.max_dict and len(self.id2w) == self.max_dict:
                    break
                self.num_vocb += 1
                self.id2w.append(token)
                self.dictionary[token] = len(self.id2w) - 1

        print('Done.')
        print('Save dictionary at %s.dict' % self.datapath)

        with open(self.datapath + '.dict', 'w+') as f:
            for token, number in self.dictionary.items():
                f.write('%s %d\n'%(token,number))

        self.index_token()
        

    def change_dict(self, dictionary):
        self.dictionary = dictionary
        self.num_vocb = len(dictionary)
        self.index_token()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.num_batch = int(len(self.sentence) / self.batch_size)
        self.index = range(self.num_batch)
        #self.describe_dataset()

    def index_token(self):
        #Convert tokens to integers
        print('Index tokens ...')
        self.sentence = []
        zero_sentence = 0
        long_sentence = 0
        with open(self.datapath, 'r') as f:
            for count, line in enumerate(f):
                
                if self.display_freq > 0 and count % self.display_freq == 0:
                    print('%d  sentence processed'%(count))
                
                tokens = line.split()
                
                if len(tokens) == 0:
                    zero_sentence += 1
                else:
                    if len(tokens) > self.max_len:
                        long_sentence += 1
                        if self.trunc_len > 0:
                            tokens = tokens[:self.trunc_len]
                        else:
                            continue
                    
                    self.num_tokens += len(tokens)
                    tokens = [const.BOS_WORD] + tokens + [const.EOS_WORD]
                    sequence = [self.dictionary[token] if token in self.dictionary else self.dictionary[const.UNK_WORD] for token in tokens]
                    self.sentence.append(sequence)
        
        self.sentence = sorted(self.sentence, key=lambda s:len(s))
        self.num_batch = int(len(self.sentence) / self.batch_size)
        self.index = range(self.num_batch)
        print('%d sentences were processed, %d longer than maximum length,%d were ignored because zero length'%(len(self.sentence), long_sentence, zero_sentence))
        self.describe_dataset()
        begin_of_group = 0
        group = []
        self.groups =[]
        for sentence in self.sentence:
            if len(sentence) <= (begin_of_group+15):
                group.append(sentence)
            else:
                if len(group) < 128:
                    self.groups[-1] += group
                else:
                    self.groups.append(group)
                begin_of_group = begin_of_group+15
                group = []
                group.append(sentence)
        self.num_batches = [int(len(group)/self.batch_size) for group in self.groups]      
        print('Done.')

        
    def get_batch_new(self, batch_idx, group_id):
        sents = self.groups[group_id]
        lengths = [len(sents[x]) for x in range(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))]
        max_len = max(lengths)
        total_len = sum(lengths)

        sorted_lengths = sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)

        batch = torch.zeros(self.batch_size, max_len, dtype=torch.long, device=device)
        target = torch.zeros(self.batch_size, max_len, dtype=torch.long, device=device)
      
        for i in range(self.batch_size):
            len_ = sorted_lengths[i][1] 
            idx_ = sorted_lengths[i][0]

            sequence_idx = idx_ + self.batch_size * batch_idx
            
            batch[i, :len_-1].copy_(torch.tensor(sents[sequence_idx][:len_-1], dtype=torch.long, device=device))
            target[i, :len_-1].copy_(torch.tensor(sents[sequence_idx][1:len_], dtype=torch.long, device=device))

        batch_lengths = torch.tensor([x[1] for x in sorted_lengths], device=device)
        
        return batch, batch_lengths, target
        
    def get_batch_random(self, batch_idx):
        batch_sents = random.sample(random.sample(self.groups, 1)[0], self.batch_size)
        
        lengths = [len(sent) for sent in batch_sents]
        max_len = max(lengths)
        total_len = sum(lengths)

        sorted_lengths = sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)

        batch = torch.zeros(self.batch_size, max_len, dtype=torch.long, device=device)
        target = torch.zeros(self.batch_size, max_len, dtype=torch.long, device=device)
      
        for i in range(self.batch_size):
            len_ = sorted_lengths[i][1] 
            idx_ = sorted_lengths[i][0]
            
            batch[i, :len_-1].copy_(torch.tensor(batch_sents[idx_][:len_-1], dtype=torch.long, device=device))
            target[i, :len_-1].copy_(torch.tensor(batch_sents[idx_][1:len_], dtype=torch.long, device=device))

        batch_lengths = torch.tensor([x[1] for x in sorted_lengths], device=device)
        
        return batch, batch_lengths, target
    
    def get_batch(self, batch_idx):
        lengths = [len(self.sentence[x]) for x in range(self.batch_size * batch_idx, self.batch_size * (batch_idx + 1))]
        max_len = max(lengths)
        total_len = sum(lengths)

        sorted_lengths = sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)

        batch = torch.zeros(self.batch_size, max_len, dtype=torch.long, device=device)
        target = torch.zeros(self.batch_size, max_len, dtype=torch.long, device=device)

        for i in range(self.batch_size):
            len_ = sorted_lengths[i][1] 
            idx_ = sorted_lengths[i][0]

            sequence_idx = idx_ + self.batch_size * batch_idx
            
            batch[i, :len_-1].copy_(torch.tensor(self.sentence[sequence_idx][:len_-1], dtype=torch.long, device=device))
            target[i, :len_-1].copy_(torch.tensor(self.sentence[sequence_idx][1:len_], dtype=torch.long, device=device))

        batch_lengths = torch.tensor([x[1] for x in sorted_lengths], device=device)

        return batch, batch_lengths, target

    def shuffle_epoch(self):
        self.groups = [random.sample(x, len(x)) for x in self.groups]
    
    def shuffle(self):
        print(self.shuffle_level)
        assert self.shuffle_level > 0, 'Enable shuffle first!'
        if self.shuffle_level == 1:
            random.shuffle(self.index)
        if self.shuffle_level == 2:
            random.shuffle(self.sentence)
    

    def change_shuffle_level(self, level):
        self.shuffle_level = level


    def __getitem__(self, index):
        if self.shuffle == 1:
            return self.get_batch(self.index[index]) 
        else:
            return self.get_batch(index)


    def __len__(self):
        return self.num_batch
    
    
def convert_text2idx(sentences, dictionary):
    return [[const.BOS] +
            [dictionary.get(token, const.UNK) for token in s] +
            [const.EOS] for s in sentences]


def convert_idx2text(ids, id2w):
    tokens = []
    for i in example:
        if i == const.EOS:
            break
        tokens.append(id2w[i])
    return ' '.join(tokens)