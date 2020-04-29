import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_mask_3d(widths, heights):
    max_w = torch.max(widths).item()
    max_h = torch.max(heights).item()
    mask = torch.zeros(widths.size(0), max_w, max_h)
    for i in range(widths.size(0)):
        mask[i,:widths[i],:heights[i]] = 1
    return mask.bool()


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
