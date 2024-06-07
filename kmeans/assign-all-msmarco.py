from sklearn.cluster import MiniBatchKMeans
import numpy
import pickle
import os
import sys
import glob
import re
import faiss
import concurrent

NUM_CLUSTERS = 1
DIM = 768
#DIM = 192
MULTI_ASSIGN = 1

def main():
    url_file = "/work/edauterman/private-search/code/embedding/embeddings_msmarco/msmarco_url.npy"
    embed_file = "/home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/embeddings.npy"

    centroids_file = ("/home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/1_law_centroids.npy")

    data = numpy.load(embed_file)

    with open(centroids_file, 'rb') as f:
        centroids = numpy.loadtxt(centroids_file)
 
    print(centroids)

    cluster_files = [("/home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/1_cluster_%d.txt" % i) for i in range(NUM_CLUSTERS)]
    assignment_dict = dict()
    kmeans = faiss.Kmeans(DIM, NUM_CLUSTERS, verbose=True, nredo=3)
    kmeans.train(data.astype(numpy.float32))
    distances, assignments = kmeans.index.search(data.astype(numpy.float32), MULTI_ASSIGN)

    print("Finished kmeans assignment")


    percentiles = []
    for i in range(1, MULTI_ASSIGN):
        percentiles.append(numpy.percentile([(dist[i] - dist[0]) for dist in distances], 20))
   
    over_assign_count = 0
    for i in range(len(assignments)):
        for k in range(MULTI_ASSIGN):
            if (k == 0) or (k > 0 and (distances[i][k] - distances[i][0]) < percentiles[k-1]):
                cluster = assignments[i][k]
                if cluster not in assignment_dict:
                    assignment_dict[cluster] = [i]
                else:
                    assignment_dict[cluster].append(i)
                if k > 0:
                    over_assign_count += 1
    

    for i in range(NUM_CLUSTERS):
        print("%d/%d" % (i, NUM_CLUSTERS))
        with open(cluster_files[i], 'w') as f:
            if i in assignment_dict:
                for idx in assignment_dict[i]:
                    embed = data[idx]
                    embstr = ",".join(["%f" % ch for ch in embed])
                    doc_id = idx
                    data_str = ("%d" % (doc_id))
                    f.write(data_str + "\n")
            else:
                print("Not in assignment dict %d" % i)

    print("Over assigning for param %d = %d" % (MULTI_ASSIGN, over_assign_count))

if __name__ == "__main__":
    main()
