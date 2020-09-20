import os
import gzip
import logging
from threading import Thread
from queue import Queue
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import fht20.data.util as util

# TODO the DB pool needs to have the appropriate size


logger = logging.getLogger("doc_puller")


def proc_doc(tokens):
    """Filters tokens to remove tokens that are whitespace only."""
    # these tokens maybe shouldn't even be there?  This could be fixed upstream in the DB.
    res = []
    for t in tokens:
        t = t.strip()
        if t == '':
            continue
        res.append(t)
    return res


class ProcessingWorker(Thread):
    def __init__(self, db, q, out_q, with_mentions):
        super(ProcessingWorker, self).__init__()
        self.with_mentions = with_mentions
        self.db = db
        self.q = q
        self.out_q = out_q

    def run(self):
        for year_and_month, docs_path, voc_path in iter(self.q.get, None):
            vocab = defaultdict(lambda: 0)
            date_slice = util.get_range_for_month(*year_and_month)
            if self.with_mentions:
                docs = self.db.get_doc_tokens_with_mentions(time_slice=date_slice)
            else:
                docs = self.db.get_doc_tokens_without_mentions(time_slice=date_slice)
            with gzip.open(docs_path, 'wb') as f:
                for doc in docs:
                    doc = proc_doc(doc)
                    line = "\t".join(doc) + "\n"
                    f.write(line.encode("UTF-8"))
                    for token in doc:
                        vocab[token] += 1
            # write vocab
            voc_tuples = sorted(vocab.items(), key=lambda i: i[1], reverse=True)
            with open(voc_path, 'w') as f:
                for row in voc_tuples:
                    line = "\t".join([f"{x}" for x in row]) + "\n"
                    f.write(line)
            # notify done
            self.out_q.put(year_and_month)


class DocPuller:
    """Pulls docs from the DB and dumps them to file for subsequent
    use."""
    def __init__(self, db, dir, n_workers,
                 start_year, start_month,
                 end_year, end_month,
                 with_mentions: bool):
        """Start and end are inclusive."""
        self.db = db
        self.dir = dir
        self.n_workers = n_workers
        self.s_y = start_year
        self.s_m = start_month
        self.e_y = end_year
        self.e_m = end_month
        self.with_mentions = with_mentions

    def create_dirs(self):
        Path(os.path.join(self.dir, 'base_data')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.dir, 'slices')).mkdir(parents=True, exist_ok=True)

    def get_paths(self, y, m):
        voc = os.path.join(self.dir, 'base_data', f'{y}-{m:02}.vocab')
        docs = os.path.join(self.dir, 'base_data', f'{y}-{m:02}.docs.gz')
        return voc, docs

    def month_exists(self, y, m):
        voc_path, docs_path = self.get_paths(y, m)
        return os.path.exists(voc_path) and os.path.exists(docs_path)

    def process(self):
        self.create_dirs()
        months = util.get_years_months(self.s_y, self.s_m, self.e_y, self.e_m)
        len_orig = len(months)
        months = [(y, m) for y, m in months if not self.month_exists(y, m)]
        len_now = len(months)
        logger.info(f"Creating vocabulary and document files for {len_now} months ({len_orig - len_now} already done)")
        q = Queue()
        out_q = Queue()
        for y, m in months:
            voc_path, docs_path = self.get_paths(y, m)
            q.put(((y, m), docs_path, voc_path))
        # start workers
        for i in range(self.n_workers):
            worker = ProcessingWorker(self.db, q, out_q, self.with_mentions)
            q.put(None)
            worker.start()
        # progress bar
        pbar = tqdm(total=len(months), desc="months processed")
        p = 0
        for y, m in iter(out_q.get, None):
            pbar.update(1)
            p += 1
            if p == len(months):
                break
