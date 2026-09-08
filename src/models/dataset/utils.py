import pyBigWig as pbw
import numpy as np
import torch
import einops
from enformer_pytorch import seq_indices_to_one_hot


def feature_to_npy(chr_name, start, end, file_path):
    """
    Robust way to read bigWig files - works both if naming is "chr1" or simply "1".
    """
    bw_file = pbw.open(file_path)

    chrom_dict = bw_file.chroms()
    start, end = int(start), int(end)
    base = chr_name[3:] if chr_name.startswith("chr") else chr_name

    if base in chrom_dict:
        key = base
    elif f"chr{base}" in chrom_dict:
        key = f"chr{base}"
    else:
        bw_file.close()
        raise KeyError(f"{chr_name} (-> tried {base!r} and {'chr' + base!r}) not in {file_path} header")
    signals = bw_file.values(key, start, end)
    bw_file.close()
    return np.array(signals)


def get_feature(path, chr_name, start, end):
    feature = feature_to_npy(chr_name, start, end, path)
    feature = np.nan_to_num(feature)  # Important! replace nan with 0
    feature = np.log(feature + 1) # log transform (as in C.Origami paper)
    return torch.tensor(feature, dtype=torch.float32).unsqueeze(0)


def get_matrix(cool, chrom, start, end, balance=False, scale=1.0):
    matrix = cool.matrix(balance=balance).fetch(f"{chrom}:{start}-{end}")
    matrix = torch.tensor(matrix, dtype=torch.float32)
    mask = torch.isfinite(matrix)
    matrix = torch.nan_to_num(matrix, nan=0.0)
    matrix = torch.log(matrix * scale + 1.0)
    return matrix, mask


def onehotencode_dna(sequence, channels=4):
    if channels == 4:
        char_to_num = {'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4}
    else:
        char_to_num = {'a': 0, 't': 1, 'c': 2, 'g': 3, 'n': 4}
    sequence = sequence.lower()
    numeric_sequence = [char_to_num[char] for char in sequence]

    # Borzoi takes 4 channels, while original C.Origami uses 5
    if channels == 4:
        one_hot_matrix = seq_indices_to_one_hot(torch.tensor(numeric_sequence)) 
        return einops.rearrange(one_hot_matrix, "l c -> c l")
    elif channels == 5:
        one_hot_matrix = np.zeros((5, len(numeric_sequence)))
        one_hot_matrix[numeric_sequence, np.arange(len(numeric_sequence))] = 1
        return torch.tensor(one_hot_matrix, dtype=torch.float32)
