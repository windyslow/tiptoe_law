from sklearn.cluster import MiniBatchKMeans
import numpy
import pickle
import os
import sys
import glob
import re
import faiss
import concurrent

#NUM_CLUSTERS = 35
#NUM_CLUSTERS = 100000 
NUM_CLUSTERS = 1
#DIM = 512
DIM = 768
MULTI_ASSIGN = 1

def main():
    url_file = "/work/edauterman/private-search/code/embedding/embeddings_msmarco/msmarco_url.npy"
    embed_file = "/home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/embeddings.npy"

    centroids_file = ("/home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/1_law_centroids.npy")

    data = numpy.load(embed_file)
    print("Loaded")
    print(data)
    kmeans = faiss.Kmeans(DIM, NUM_CLUSTERS, verbose=True, nredo=3)
    kmeans.train(data.astype(numpy.float32))
    centroids = kmeans.centroids
    print(centroids)
    numpy.savetxt(centroids_file, centroids)


    print("Finished kmeans find centroids")

if __name__ == "__main__":
    main()
