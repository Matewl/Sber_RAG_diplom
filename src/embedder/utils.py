import numpy as np

def prepare_embs(emb):
    emb = np.array(emb)
    emb /= (((emb**2).sum(1))**0.5).reshape((emb.shape[0], 1))
    return emb

def prepare_emb(emb):
    emb = np.array(emb)
    emb /= ((emb**2).sum())**0.5
    return emb  
