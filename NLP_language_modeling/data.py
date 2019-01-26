import os
import torch

from collections import Counter


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
    def __init__(self, path, dictionary=None):
        """
        :param path: path to train, val, or test data
        :param dictionary: If None, create new dictionary. Else, use the given dictionary.
        """
        self.dictionary = Dictionary() if dictionary is None else dictionary
        self.new_dict = True if dictionary is None else False
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if self.new_dict:  # if building a new dictionary, add all the new words you come across
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>'])
                    token += 1

        return ids


class CorpusWikiTextChar(object):
    def __init__(self, path, dictionary):
        """
        :param path: path to train, val, or test data
        :param dictionary: If None, create new dictionary. Else, use the given dictionary.
        """
        self.dictionary = dictionary
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        corpus = []
        ids = []
        with open(path, 'r') as f:
            for line in f:
                if len(line) == 1:  # end of example
                    continue

                words = line.split()
                for i in range(len(words)):
                    word = words[i]
                    word = word.lower()
                    for char in [char for char in word]:
                        if char in [0,1,2,3,4,5,6,7,8,9]:
                            char = 'N'
                        if char not in self.dictionary.word2idx.keys():
                            continue # don't append it to the corpus
                        corpus.append(char)
                        ids.append(self.dictionary.word2idx[char])

                    if i < len(words) - 1:
                        corpus.append('_')
                        ids.append(self.dictionary.word2idx['_'])
                corpus.append('<eos>')
                ids.append(self.dictionary.word2idx['<eos>'])

        return torch.LongTensor(ids)


class OODCorpus(object):
    def __init__(self, path, dictionary, char=False):
        """
        :param path: path to train, val, or test data
        :param dictionary: existing dictionary of words constructed with Corpus class on in-dist
        :param char: if True, return character-level data
        """
        self.dictionary = dictionary
        self.data_words, self.data = self.tokenize(path, char)

    def tokenize(self, path, char=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        corpus = []
        ids = []
        with open(path, 'r') as f:
            for line in f:
                if len(line) == 1:  # end of example
                    if char:
                        corpus.append('<eos>')
                        ids.append(self.dictionary.word2idx['<eos>'])
                    else:
                        corpus.append('<eos>')
                        ids.append(self.dictionary.word2idx['<eos>'])
                    continue
                word = line.split('\t')[1]
                if char:
                    if word not in self.dictionary.word2idx.keys():
                        word = '<unk>'
                    corpus.extend(list(word))
                    corpus.append('_')
                    ids.extend([self.dictionary.word2idx[char] for char in word])
                    ids.append(self.dictionary.word2idx['_'])
                else:
                    corpus.append(word)
                    ids.append(self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>']))

        return corpus, torch.LongTensor(ids)
