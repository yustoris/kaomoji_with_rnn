from collections import defaultdict
import numpy as np

class DataLoader:
    def __init__(self):
        self.chars = set()
        lines = []
        for l in open('./kaomojis'):
            self.chars |= set(l[:-1])
            lines.append(l[:-1])

        self.char_to_idx = { c:i for i,c in enumerate(self.chars) }
        self.idx_to_char = { i:c for i,c in enumerate(self.chars) }
            
        self.char_vecs = []
        words_num = len(self.chars)
        for line in lines:
            char_vec = [self.char_to_idx[c] for c in line]
            input_vec = [self._one_hot_vec(words_num+1, idx) for idx in char_vec]            
            output_vec = [self._one_hot_vec(words_num+1, idx) for idx
                          in char_vec[1:] + [words_num]]
            self.char_vecs.append((input_vec, output_vec))

    def _one_hot_vec(self, length, char_idx):
        vec = np.zeros((length, 1))
        vec[char_idx] = 1.0
        return vec


