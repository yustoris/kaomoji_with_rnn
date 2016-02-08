from collections import defaultdict
import numpy as np

class DataLoader:
    def __init__(self):
        self.chars = set()
        lines = []
        for l in open('./kaomojis_tiny'):
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
# from collections import defaultdict
# import pprint
# import numpy as np

# class DataLoader:
#     def __init__(self):
#         c = 0
#         table = dict()
#         tr = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ!?,.:\"\'()', 'abcdefghijklmnopqrstuvwxyz         ')
#         english_sentences = []
#         self.english_words_bag = defaultdict(int)
#         taken = set()
#         for l in open('corpus_en'):
#             if len(l[:-1].split(' ')) <= 10:
#                 sentence = []
#                 for w in l[:-1].split(' '):
#                     ww = w.translate(tr).strip()
#                     self.english_words_bag[ww]+=1
#                     sentence.append(ww)
#                 english_sentences.append(sentence)
#                 taken.add(c)
#             c+=1

#         # pprint.pprint(len(english_words_bag.keys()))
#         # pprint.pprint(self.english_words_bag.keys())
#         print(len(english_sentences))
#         c = 0
#         self.french_words_bag = defaultdict(int)
#         french_sentences = []
#         for l in open('corpus_fr'):
#             if c in taken:
#                 sentence = []                
#                 for w in l.split(' '):
#                     ww = w.translate(tr).strip()
#                     self.french_words_bag[ww]+=1
#                     sentence.append(ww)                    
#                 french_sentences.append(sentence)                    
#             c+=1

#         self.english_words_m = dict()
#         for i,w in enumerate(self.english_words_bag.keys()):
#             self.english_words_m[w] = i

#         self.french_words_m = dict()
#         for i,w in enumerate(self.french_words_bag.keys()):
#             self.french_words_m[w] = i
#         self.ml = max(len(self.english_words_bag.keys()), len(self.french_words_bag.keys()))
            
#         self.vectors = []
#         for e, f in zip(english_sentences, french_sentences):
#             ev = []
#             for w in e:
#                 ev.append(self.english_words_m[w])
                        
#             fv = []
#             for w in f:
#                 fv.append(self.french_words_m[w])

#             if len(ev) > len(fv):
#                 for i in range(len(ev)-len(fv)):
#                     fv.append(self.ml+1)
#             elif len(fv) > len(ev):
#                 for i in range(len(fv)-len(ev)):
#                     ev.append(self.ml+1)
                    
#             input_v = self._one_hot_vec(self.ml, ev)
#             output_v = self._one_hot_vec(self.ml, fv)
#             self.vectors.append(
#                 (np.array(input_v),
#                 np.array(output_v))
#             )
            
#     def _one_hot_vec(self, mll, vec):
#         return [ [1.0 if j+1 == i else 0.0 for j in range(mll+1)] for i in vec]


# if __name__ == '__main__':
#     dl = DataLoader()

