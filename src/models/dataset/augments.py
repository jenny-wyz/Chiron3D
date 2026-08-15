import torch


def reverse_complement(sequence, features, matrix, chance=0.5):
    """Reverse complements sequence, features, and matrix half of the time"""
    if torch.rand(1).item() >= chance:
        return sequence, features, matrix

    # Reverse sequence (5xN or 6xN with motif mask)
    sequence_r = torch.flip(sequence, dims=[0])

    # TODO: I dont use features for now, but potential bug. I should make sure features are in same shape as DNA
    # Reverse features
    features_r = torch.flip(features, dims=[1]) if features is not None else None

    # Reverse hic
    matrix_r = torch.flip(matrix, dims=[0, 1])

    seq_rc = torch.zeros_like(sequence_r)

    seq_rc[:, 0] = sequence_r[:, 3]  # A -> T
    seq_rc[:, 3] = sequence_r[:, 0]  # T -> A
    seq_rc[:, 1] = sequence_r[:, 2]  # C -> G
    seq_rc[:, 2] = sequence_r[:, 1]  # G -> C

    return seq_rc, features_r, matrix_r


def gaussian_noise(inputs, std=0.1):
    """Add Gaussian noise to a PyTorch tensor."""
    noise = torch.randn_like(inputs) * std
    return inputs + noise


def shift_aug(chrom_len, start, end, resolution, flank=0, shift_bins=9):
    """Shift the TARGET region by a random number of resolution-sized bins.

    `flank` is the bp of Borzoi context needed on each side of [start, end);
    the shift is clamped so the full input window stays inside the chromosome.
    """
    region_length = end - start

    max_shift_left = (start - flank) // resolution
    max_shift_right = (chrom_len - flank - end) // resolution

    lo = -min(shift_bins, max_shift_left)
    hi = min(shift_bins, max_shift_right)
    if hi < lo:
        return start, end

    shift = torch.randint(low=int(lo), high=int(hi) + 1, size=(1,)).item()
    new_start = start + shift * resolution
    return new_start, new_start + region_length
