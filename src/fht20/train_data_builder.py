import copy
import json
import logging
import gzip
import os
import re
from functools import partial
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import scipy.sparse as ss
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import fht20.data.util as util


logger = logging.getLogger("train")


def get_pmi_info(args):
    window_size, index_lookup, tokenss = args
    voc_size = len(index_lookup)
    token_counts = np.zeros(voc_size)
    coocc_counts = defaultdict(lambda: defaultdict(lambda: 0))
    for tokens in tokenss:
        for i in range(len(tokens)):
            token = tokens[i]
            token_id = index_lookup.get(token, None)
            if not token_id:
                continue
            token_counts[token_id] += 1
            context = tokens[max(0, i - window_size):i] + \
                      tokens[i+1:min(len(tokens), i+1+window_size)]
            for c in context:
                c_id = index_lookup.get(c, None)
                if c_id:
                    coocc_counts[token_id][c_id] += 1
    coocc_counts = {k: dict(v) for k, v in coocc_counts.items()}
    return token_counts, coocc_counts


def sum_2d_dicts(dicts: List[Dict[int, Dict[int, int]]]):
    res = defaultdict(lambda: defaultdict(lambda: 0))
    for d in dicts:
        for i1, others in d.items():
            for i2, v in others.items():
                res[i1][i2] += v
    res = {k: dict(v) for k, v in res.items()}
    return res


def get_PMI_matrix(token_countss, coocc_countss):
    """Returns a sparse PMI matrix."""
    token_counts = np.array(token_countss).sum(axis=0)
    voc_size = len(token_counts)
    coocc_counts = sum_2d_dicts(coocc_countss)
    total_token_count = token_counts.sum()

    m = ss.lil_matrix((voc_size, voc_size))
    for i1 in tqdm(range(voc_size), desc="PMI", miniters=100):
        for i2 in range(voc_size):
            v = coocc_counts[i1][i2]
            if v == 0:
                continue
            pmi = np.log((v * total_token_count)/\
                         (token_counts[i1] * token_counts[i2]))
            if pmi <= 0:
                continue
            m[i1, i2] = pmi
    m = ss.csr_matrix(m)
    return m


def get_PMI_matrix_par_helper(args):
    token_counts, coocc_counts, voc_size, total_token_count, indices = args

    m = ss.lil_matrix((voc_size, voc_size))
    for i1 in indices:
        if i1 not in coocc_counts:
            continue
        for i2 in range(voc_size):
            v = coocc_counts[i1].get(i2, None)
            if not v:
                continue
            pmi = np.log((v * total_token_count)/\
                         (token_counts[i1] * token_counts[i2]))
            if pmi <= 0:
                continue
            m[i1, i2] = pmi
    m = ss.csr_matrix(m)
    return m


def get_PMI_matrix_parallel(token_countss, coocc_countss, n_processes, pool, desc):
    """Returns a sparse PMI matrix."""
    token_counts = np.array(token_countss).sum(axis=0)
    voc_size = len(token_counts)
    coocc_counts = sum_2d_dicts(coocc_countss)
    total_token_count = token_counts.sum()

    batch_size = int(voc_size / n_processes) + 1
    batches = util.batch_list(list(range(voc_size)), batch_size)
    
    args = [(token_counts, coocc_counts,
             voc_size, total_token_count,
             batch)
            for batch in batches]

    ms = []
    pbar = tqdm(total=len(args), desc=desc)
    for m in pool.map(get_PMI_matrix_par_helper, args):
        ms.append(m)
        pbar.update(1)
    m = sum(ms)
    return m


def save_PMI_matrix(m, out_path):
    with open(out_path, 'wb') as f:
        ss.save_npz(f, m)
    

def load_vocab(tsv_file) -> Counter:
    """Takes a tsv file.  The file should have two columns: tokens, counts.
    The result is a Counter.  The Counters can be summed easily."""
    with open(tsv_file, 'r') as f:
        lines = f.read().strip().split('\n')
    tuples = [l.split('\t') for l in lines]
    for i, t in enumerate(tuples):
        if len(t) != 2:
            print(f"{tsv_file}: {i}: {t}")
    tuples = [(t[0], int(t[1])) for t in tuples if len(t) == 2]
    return Counter(dict(tuples))

            
def get_sorted_vocab_with_cutoff(vocab_counter, min_count):
    tuples = [(w, c) for w, c in vocab_counter.items()
              if c >= min_count]
    index = sorted(tuples, key=lambda i: i[0])
    reverse_lookup = {w: i for i, w in enumerate([w for w, _ in index])}
    return index, reverse_lookup


class TrainDataBuilder:
    def __init__(self, data_dir, out_dir, window_size, min_count,
                 start_year, start_month, slice_size, slice_count):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.min_count = min_count
        self.window_size = window_size
        self.pmis = []
        self.start_year = start_year
        self.start_month = start_month
        self.slice_size = slice_size
        self.slice_count = slice_count
        self.blocks = util.get_years_month_blocks(start_year, start_month, slice_count, slice_size)
        # the absolute range we look at; used to find the vocabulary
        self.total_start = self.blocks[0][0]
        self.total_end = self.blocks[-1][-1]

    @property
    def _info_file_path(self):
        return os.path.join(self.out_dir, 'info.json')

    @property
    def _vocab_file_path(self):
        return os.path.join(self.out_dir, 'vocab.tsv')

    def _write_info_file(self):
        info = {
            'slice_count': self.slice_count,
            'window_size': self.window_size,
            'min_count': self.min_count
        }
        with open(self._info_file_path, 'w') as f:
            json.dump(info, f, indent=4)

    def _update_info_file(self, vocab_size):
        with open(self._info_file_path, 'r') as f:
            data = json.load(f)
        data['vocab_size'] = vocab_size
        with open(self._info_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _load_index_lookup(self) -> Dict[str, int]:
        """Loads the dictionary mapping tokens to IDs from file."""
        index_lookup = {}
        with open(self._vocab_file_path, 'r') as f:
            for i, line in enumerate(f.read().strip().split('\n')):
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                token = parts[0]
                index_lookup[token] = i
        return index_lookup

    def process_block(self, index_lookup, i, n_processes):
        block = self.blocks[i]
        out_path = os.path.join(self.out_dir, "ppmi_matrices", f"{i:02}.npz")
        if os.path.exists(out_path):
            logger.info(f"Matrix {i} exists already, skipping.")
            return
        token_countss = []
        coocc_countss = []
        with ProcessPoolExecutor(n_processes) as p:
            for month_i, item in enumerate(block):
                y, m = item
                path = os.path.join(self.data_dir, f"{y}-{m:02}.docs.gz")
                with gzip.open(path, 'r') as f:
                    content = f.read().decode('UTF-8')
                docs = content.strip().split('\n')
                tokenss = [doc.split('\t') for doc in docs]
                # divide in batches an exec in pool and call 'get_pmi_info'
                batch_size = int(len(tokenss) / n_processes) + 1
                batches = list(util.batch_list(tokenss, batch_size))
                desc = f"PMI Info Batches (Block {i+1}/{self.slice_count}, Month {month_i+1}/{len(block)})"
                pbar = tqdm(total=len(batches), desc=desc)
                args = [(self.window_size, index_lookup, batch)
                        for batch in batches]
                for tc, cc in p.map(get_pmi_info, args):
                    token_countss.append(tc)
                    coocc_countss.append(cc)
                    pbar.update(1)
            desc = f"PPMI (Block {i+1}/{self.slice_count})"
            pmi_matrix = get_PMI_matrix_parallel(token_countss, coocc_countss, n_processes, p, desc)
        save_PMI_matrix(pmi_matrix, out_path)

    def collect_all(self, n_processes):
        """Creates a training directory with the data inside.
        First creates vocabulary, then creates PPMI matrices.
        If any file is there already, the step is skipped."""
        logger.info("--==<< S E T U P   &   I N F O   F I L E >>==--")
        # ensure directories exist
        out_dir_path = Path(self.out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        matrix_dir = Path(os.path.join(self.out_dir, "ppmi_matrices"))
        matrix_dir.mkdir(parents=True, exist_ok=True)
        models_dir = Path(os.path.join(self.out_dir, "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        # create info file if it's not there already
        if not os.path.exists(self._info_file_path):
            self._write_info_file()
        logger.info("--==<<        V O C A B U L A R Y        >>==--")
        years_months = util.get_years_months(*self.total_start, *self.total_end)
        if not os.path.exists(self._vocab_file_path):
            logger.info("No vocabulary file found.")
            vocab_files = [os.path.join(self.data_dir, f"{y}-{m:02}.vocab")
                           for y, m in years_months]
            logger.info(f"{len(vocab_files)} monthly vocabularies will be scanned and a cutoff of {self.min_count} will be applied.")
            total_vocab = Counter()
            for voc_file in tqdm(vocab_files, desc="Scanning vocab files ..."):
                total_vocab += load_vocab(voc_file)
            index, index_lookup = get_sorted_vocab_with_cutoff(total_vocab, self.min_count)
            self._update_info_file(len(index))
            util.write_tsv_file(index, self._vocab_file_path)
            del index
        else:  # file exists
            logger.info("Vocabulary file already created.  Loading ...")
            index_lookup = self._load_index_lookup()
        vocab_size = len(index_lookup)
        logger.info(f"Vocabulary size: {vocab_size:,}")
        logger.info("--==<<     P P M I   M A T R I C E S     >>==--")
        logger.info(f"{self.slice_count} PPMI matrices need to be calculated.  Each covering {self.slice_size} month(s).")
        for i in tqdm(range(self.slice_count), desc="PPMI matrices"):
            self.process_block(copy.deepcopy(index_lookup), i, n_processes)
        logger.info("--==<<            D  O  N  E             >>==--")


def train_data_builder_from_str(data_str, data_dir, out_dir) -> TrainDataBuilder:
    name, date, slice_info, pmi_info = data_str.split('_')
    start_year, start_month = [int(x) for x in date.split('-')]
    slice_count, slice_size = [int(x) for x in slice_info.split('x')]
    window_size = int(re.findall('w(\d+)', pmi_info)[0])
    min_count = int(re.findall('m(\d+)', pmi_info)[0])
    return TrainDataBuilder(data_dir, out_dir, window_size, min_count,
                            start_year, start_month, slice_size, slice_count)
