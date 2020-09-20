import json
import os
import numpy as np
from functools import cached_property
from gensim.models.keyedvectors import KeyedVectors
from scipy.spatial import distance
from typing import Dict, List, Tuple

import fht20.data.util as util


def get_counts_for_words(word_counts: Dict[str, int], words: List[str]):
    """Given a vector (list) for words, and a dictionary of word counts,
    returns a numpy array of the same size as 'words' containing the counts
    for the words."""
    counts = np.zeros(len(words))
    for i, word in enumerate(words):
        c = word_counts.get(word, None)
        if c:
            counts[i] = c
    return counts


class DataInfo:
    def __init__(self, path):
        self.path = path
        
    def get_vocab_with_counts(self, year, month):
        """Given a year and a month, returns a dictionary
        mapping words to word counts for that time period."""
        p = os.path.join(self.path, "base_data", f"{year}-{month:02}.vocab")
        with open(p, 'r') as f:
            lines = f.read().strip().split('\n')
        rows = [l.split('\t') for l in lines]
        res = [(r[0], int(r[1])) for r in rows]
        return dict(res)
    
    def get_counts_for_words_for_months(self, words: List[str],
                                        months: List[Tuple[int, int]]):
        counts = np.zeros(len(words))
        for year, month in months:
            word_counts = self.get_vocab_with_counts(year, month)
            counts += get_counts_for_words(word_counts, words)
        return counts
    
    def get_vocab_counts_for_blocks(self, vocab: List[str], blocks):
        res = []
        for block in blocks:
            res.append(self.get_counts_for_words_for_months(vocab, block))
        return np.vstack(res).T


class SliceInfo:
    def __init__(self, path):
        data_dir = path.split('/slices/')[0]
        self.data_info = DataInfo(data_dir)
        self.info_path = os.path.join(path, 'info.json')
        self.vocab_path = os.path.join(path, 'vocab.tsv')
        
    @cached_property
    def _info_json(self):
        with open(self.info_path, 'r') as f:
            return json.load(f)
    
    @property
    def slice_count(self):
        return self._info_json['slice_count']
    
    @cached_property
    def vocab_tokens(self):
        with open(self.vocab_path, 'r') as f:
            lines = f.read().strip().split('\n')
        tokens = [l.split('\t')[0] for l in lines]
        return tokens
    
    @cached_property
    def vocab_counts(self):
        with open(self.vocab_path, 'r') as f:
            lines = f.read().strip().split('\n')
        rows = [l.split('\t') for l in lines]
        x = [(r[0], int(r[1])) for r in rows]
        return dict(x)
    
    @cached_property
    def _vocab_counts_over_time(self):
        """A V x T matrix containing the word counts."""
        blocks = util.get_years_month_blocks(2000, 1, 20, 12)
        return self.data_info.get_vocab_counts_for_blocks(self.vocab_tokens, blocks)
    
    @cached_property
    def counts(self):
        """A dictionary mapping words to vectors of length T."""
        return dict(zip(self.vocab_tokens,
                        self._vocab_counts_over_time))
    
    @cached_property
    def total_counts(self):
        return self._vocab_counts_over_time.sum(axis=0)
    
    def normalized_counts(self, word: str):
        """Returns a vector of length T.  Contains word counts, normalized
        by absolute token count for time t."""
        cs = self.counts[word]
        return cs / self.total_counts


class DiachronicEmbeddings:
    def __init__(self, vocab, embeddings):
        """embeddings should be a list of 2D arrays."""
        self.T = len(embeddings)
        self.V, self.d = embeddings[0].shape
        self.vocab = vocab
        self.embeds = []
        for t in range(self.T):
            kv = KeyedVectors(self.d)
            kv.add(vocab, embeddings[t])
            self.embeds.append(kv)
            
    def get_embeddings(self, token):
        """Returns a list of length T with the embedding vectors of length d."""
        return [e[token] for e in self.embeds]

    def get_embeddings_for_tokens_from_slice(self, slice_index: int, tokens: List[str]):
        """Takes an index of a slice and a list of tokens and returns a map,
        mapping the tokens to their embeddings in that slice.
        If tokens is None, return all."""
        if tokens is None:
            tokens = self.vocab
        else:
            v = set(self.vocab)
            tokens = [t for t in tokens if t in v]
        return {t: self.embeds[slice_index][t] for t in tokens}
    
    def get_cosine_diffs(self, token):
        embeds = self.get_embeddings(token)
        diffs = []
        for t1, t2 in zip(embeds[:-1], embeds[1:]):
            diffs.append(distance.cosine(t1, t2))
        return np.array(diffs)
    
    def get_change_indices(self, token, f):
        """Returns a list of change indices for the given token."""
        diffs = self.get_cosine_diffs(token)
        mean = diffs.mean()
        std_dev = diffs.std()
        res = [i+1 for i in range(len(diffs)) if diffs[i] > (mean + f * std_dev)]
        return res

    def get_biggest_change_index(self, token):
        diffs = self.get_cosine_diffs(token)
        return diffs.argmax() + 1
    
    def get_change_indices_many(self, tokens, f):
        res = {}
        for t in tokens:
            ci = self.get_change_indices(t, f)
            res[t] = ci
        return res
    
    def get_threshold_change_indices(self, token, threshold):
        diffs = self.get_cosine_diffs(token)
        res = [i+1 for i in range(len(diffs)) if diffs[i] >= threshold]
        return res
    
    def get_many_threshold_change_indices(self, tokens, threshold):
        res = {}
        for t in tokens:
            ci = self.get_threshold_change_indices(t, threshold)
            res[t] = ci
        return res
        
    
    # TODO implement a function that returns changes above a certain threshold (absolute)


class ModelInfo:
    def __init__(self, path):
        slice_path, _ = path.split('/models/')
        self.slice_info = SliceInfo(slice_path)
        self.path = path
        
    def _latest_epoch_dir(self):
        d = sorted(os.listdir(os.path.join(self.path, 'epochs')))[-1]
        return os.path.join(self.path, 'epochs', d)
        
    def _load_latest_epoch(self):
        """Returns a list of T embeddings, one set per time slice.
        The embeddings are created by concatenating U and V."""
        res = []
        latest_e_d = self._latest_epoch_dir()
        for t in range(self.slice_info.slice_count):
            p_u = os.path.join(latest_e_d, f"u{t:02}.npz")
            u = np.load(p_u)['data']
            p_v = os.path.join(latest_e_d, f"v{t:02}.npz")
            v = np.load(p_v)['data']
            concatted = np.hstack((u, v))
            res.append(concatted)
        return res

    @property
    def vocab(self):
        return self.slice_info.vocab_tokens
    
    @cached_property
    def embeddings(self):
        """The embeddings trained in the latest epoch."""
        return DiachronicEmbeddings(self.vocab,
                                    self._load_latest_epoch())
