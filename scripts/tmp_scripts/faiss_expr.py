import faiss
import numpy as np


def create_vectors():
    d = 64  # dimension
    nb = 100000  # database size
    nq = 10000  # nb of queries
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    return xb, xq, d


def index_vectors(xb, d):
    index = faiss.IndexFlatIP(d)  # build the index
    print(index.is_trained)
    index.add(xb)  # add vectors to the index
    print(index.ntotal)
    return index, xb


def search_faiss(index, xb, xq):
    k = 4  # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k)  # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries


def index_and_search():
    xb, xq, d = create_vectors()
    index, xb = index_vectors(xb, d)
    search_faiss(index, xb, xq)


if __name__ == '__main__':
    index_and_search()
