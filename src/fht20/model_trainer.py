import json
import logging
import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as ss
from functools import cached_property
from pathlib import Path
from typing import List
from tqdm import tqdm


logger = logging.getLogger()


def frob(M):
    """Calculates the squared Frobenius norm of a matrix."""
    return np.square(abs(M)).sum()


class Vocabulary:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.tokens = [t.split('\t')[0]
                           for t in f.read().strip().split('\n')]
        self.ids = {}
        for i, t in enumerate(self.tokens):
            self.ids[t] = i


class TrainingData:
    """A representation of the training data for the model.
    The training data is in a directory with an info.json.
    There is a subdirectory ppmi_matrices that contains .tsv files
    which contain the PPMI matrices."""
    def __init__(self, directory):
        self.directory = directory
        
    @cached_property
    def time_slice_count(self):
        with open(os.path.join(self.directory, 'info.json'), 'r') as f:
            data = json.load(f)
            return data['slice_count']

    @cached_property
    def vocab(self):
        return Vocabulary(os.path.join(self.directory, 'vocab.tsv'))
        
    @cached_property
    def vocab_size(self):
        with open(os.path.join(self.directory, 'info.json'), 'r') as f:
            data = json.load(f)
            return data['vocab_size']
        
    def load_ppmi_matrix(self, i: int):
        """Loads a PPMI matrix by index. self.time_slice_count gives the number of slices.
        Supports loading from npz and tsv files."""
        npz_file = os.path.join(self.directory, "ppmi_matrices", f"{i:02}.npz")
        tsv_file = os.path.join(self.directory, "ppmi_matrices", f"{i:02}.tsv")
        if os.path.exists(npz_file):
            ppmi_matrix = ss.load_npz(npz_file)
        elif os.path.exists(tsv_file):
            data = pd.read_csv(tsv_file, sep='\t').values
            ppmi_matrix = ss.coo_matrix((data[:,2],(data[:,0],data[:,1])),
                                        shape=(self.vocab_size, self.vocab_size))
        else:
            raise Exception("Data for time slice {i} is missing!")
        ppmi_matrix = ss.csr_matrix(ppmi_matrix)  # convert for performance
        return ppmi_matrix


class EmbeddingTrainer:
    def __init__(self, training_data_dir, run_name,
                 embedding_dim, gam, lam, tau, batch_size):
        self.training_data = TrainingData(training_data_dir)
        self.T = self.training_data.time_slice_count
        self.run_dir = os.path.join(training_data_dir, "models", run_name)
        self.embedding_dim = embedding_dim
        self.gam, self.lam, self.tau = gam, lam, tau
        self.batch_size = batch_size
        self.U, self.V = [], []
        
    def init_run_dir(self):
        """Creates all the required directories if they are missing."""
        run_dir_path = Path(self.run_dir)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.run_dir, "epochs")).mkdir(exist_ok=True)
        
    def init_info_file(self):
        """Initializes an info file with the hyper paramters of the model
        and other settings."""
        p = os.path.join(self.run_dir, 'info.json')
        if os.path.exists(p):
            return  # file exists already
        data = {
            'gam': self.gam,
            'lam': self.lam,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'embedding_dim': self.embedding_dim,
            'training_data_dir': self.training_data.directory
        }
        with open(p, 'w') as f:
            json.dump(data, f)
        
    def epochs_so_far(self):
        """Returns the number of epochs saved to disk so far.
        Epoch 0 is the initialization."""
        files = os.listdir(os.path.join(self.run_dir, "epochs"))
        return len(files)
    
    def get_batch_indices_randomized(self, n: int) -> List[List[int]]:
        """Takes the total number of samples and generates batches.
        A batch is a list of indices."""
        indices = np.random.permutation(n)
        batches = np.split(indices, range(self.batch_size, n,
                                          self.batch_size))
        return batches
        
    def init_embeddings(self):
        """Initialize U and V with random matrices."""
        V = self.training_data.vocab_size
        r = self.embedding_dim
        self.U.append(np.random.randn(V, r)/np.sqrt(r))
        self.V.append(np.random.randn(V, r)/np.sqrt(r))
        for t in range(self.training_data.time_slice_count - 1):
            self.U.append(self.U[0].copy())
            self.V.append(self.V[0].copy())
        self.save_embeddings(0)  # save as zeroth epoch
        
    def save_embeddings(self, epoch: int):
        path = os.path.join(self.run_dir, 'epochs', f"{epoch:02}")
        Path(path).mkdir(parents=True, exist_ok=True)
        pbar = tqdm(total=self.T * 2, desc="Saving", leave=False)
        for i in range(self.T):
            np.savez(os.path.join(path, f"u{i:02}.npz"), data=self.U[i])
            pbar.update(1)
            np.savez(os.path.join(path, f"v{i:02}.npz"), data=self.V[i])
            pbar.update(1)
        
    def load_embeddings(self, epoch: int):
        base_path = os.path.join(self.run_dir, 'epochs', f"{epoch:02}")
        pbar = tqdm(total=self.T * 2, desc="Loading", leave=False)
        for t in range(self.T):
            self.U.append(np.load(os.path.join(base_path, f"u{t:02}.npz"))['data'])
            pbar.update(1)
            self.V.append(np.load(os.path.join(base_path, f"v{t:02}.npz"))['data'])
            pbar.update(1)

    def log_error(self, error_dict):
        """Takes a dictionary of error information, and appends it to a json file."""
        error_file = os.path.join(self.run_dir, 'errors.json')
        errors = []
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                errors = json.load(f)
        errors.append(error_dict)
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=4)
    
    def _get_empty_embeddings(self):
        V = self.training_data.vocab_size
        r = self.embedding_dim
        return np.zeros((V, r))
        
    def validate(self, Y, U, V):
        n = Y.shape[0]
        error = 0
        batches = self.get_batch_indices_randomized(n)
        for inds in tqdm(batches, desc="Validation Batches", leave=False):
            Y_b = Y[inds, :].todense()
            error += frob(Y_b - np.dot(U[inds], V.T))
        return error

    def val(self):
        E1 = []  # the error the the ppmi matrix
        pbar = tqdm(total=self.T * 4 - 1, desc="Calculating Error")
        for t in range(self.T):
            Y = self.training_data.load_ppmi_matrix(t)
            E1.append(0.5 * self.validate(Y, self.U[t], self.V[t]))
            pbar.update(1)
        E_gam = []
        for t in range(self.T):
            E_gam.append((self.gam/2) * frob(self.U[t] - self.V[t]))
            pbar.update(1)
        E_lam_U = []
        E_lam_V = []
        for t in range(self.T):
            E_lam_U.append((self.lam/2) * frob(self.U[t]))
            E_lam_V.append((self.lam/2) * frob(self.V[t]))
            pbar.update(1)
        E_tau_U = []
        E_tau_V = []
        for t in range(1, self.T):
            E_tau_U.append((self.tau/2) * frob(self.U[t-1] - self.U[t]))
            E_tau_V.append((self.tau/2) * frob(self.V[t-1] - self.V[t]))
            pbar.update(1)
        total = sum(E1) + sum(E_gam) + sum(E_lam_U) + sum(E_lam_V) + sum(E_tau_U) + sum(E_tau_V)
        errors = {
            'E_y': E1,
            'E_gam': E_gam,
            'E_lam_U': E_lam_U,
            'E_lam_V': E_lam_V,
            'E_tau_U': E_tau_U,
            'E_tau_V': E_tau_V,
            'total': total
        }
        return errors
        
    def update_batch_inds(self, U, Y, V_p, V_n, is_edge: bool, inds):
        f = 1 if is_edge else 2
        d = U.shape[1]  # the embedding dim
        A = np.dot(U.T, U) + (self.lam + f*self.tau + self.gam) * np.eye(d)
        Yb = Y[inds,:].todense()
        Uty = np.dot(Yb, U)
        B = Uty + self.gam*U[inds, :] + self.tau*(V_p[inds, :] + V_n[inds, :])
        Vhat = np.linalg.lstsq(A, B.T, rcond=None)
        return Vhat[0].T
        
    def epoch(self):
        """An epoch means that every embedding in U and V is updated once.
        Afterwards the new embeddings are also written to file."""
        T = self.training_data.time_slice_count
        steps = np.random.permutation(T)
        for t in tqdm(steps, desc="Running epoch, timesteps"):
            Y = self.training_data.load_ppmi_matrix(t)
            is_edge = t == 0 or t == T-1
            
            U_p = self.U[t-1] if t > 0 else self._get_empty_embeddings()
            U_n = self.U[t+1] if t < T - 1 else self._get_empty_embeddings()
            V_p = self.V[t-1] if t > 0 else self._get_empty_embeddings()
            V_n = self.V[t+1] if t < T - 1 else self._get_empty_embeddings()
            
            n = Y.shape[0]
            batches = self.get_batch_indices_randomized(n)
            for inds in tqdm(batches, desc="Batches", leave=False):
                self.V[t][inds, :] = self.update_batch_inds(self.U[t], Y, 
                                                            V_p, V_n,
                                                            is_edge, inds)
                self.U[t][inds, :] = self.update_batch_inds(self.V[t], Y,
                                                            U_p, U_n,
                                                            is_edge, inds)
        #error_info = self.val()
        #self.log_error(error_info)
        #print(f"Epoch error: {error_info['total']:,}")
            
    def run(self, epochs: int):
        self.init_run_dir()
        self.init_info_file()
        epochs_so_far = self.epochs_so_far()
        if epochs_so_far == 0:
            # no saved states, initialize random and save as 0th epoch
            self.init_embeddings()
            # increase counter, because the 0th epoch is there now
            epochs_so_far += 1
        else:
            self.load_embeddings(epochs_so_far - 1)  # load most recent
        for epoch in tqdm(range(epochs_so_far, epochs+1), desc="Epochs"):
            self.epoch()
            self.save_embeddings(epoch)


def trainer_from_str(training_data_dir, model_str, batch_size) -> None:
    _, embed_dim, params = model_str.split('_')
    embed_dim = int(embed_dim)
    lam = int(re.findall('l(\d+)', params)[0])
    tau = int(re.findall('t(\d+)', params)[0])
    gam = int(re.findall('g(\d+)', params)[0])
    return EmbeddingTrainer(training_data_dir, model_str,
                            embed_dim, gam, lam, tau, batch_size)
    
