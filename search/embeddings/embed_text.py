import sys
import numpy
import time
import json
from transformers import AutoTokenizer, AutoModel

# New size for the embeddings
NUM_CLUSTERS = 1
new_dimension = 192
prec = 5

CENTROIDS_FILE = "%s/artifact/dim192/index.faiss"
PCA_COMPONENTS_FILE = "%s/artifact/dim192/pca_192.npy"

def find_nearest_clusters(cluster_index, query, num_clusters):
        query_float = numpy.array(query).astype('float32')
        results = cluster_index.search(query_float, NUM_CLUSTERS)
        for i in range(len(results[0][0])):
            cluster_id = results[1][0][i]
            if cluster_id < num_clusters:
                return cluster_id
            #print("dist: %d id: %d" % (results[0][0][i], results[1][0][i]))
        return 0

def find_nearest_clusters_from_file(centroids, query_embed, num_clusters):
    query_float = numpy.array(query_embed).astype('float32')
    distances = numpy.asarray(numpy.matmul(centroids, numpy.transpose(numpy.asmatrix(query_float))))
    res = numpy.argpartition(distances, -NUM_CLUSTERS, axis=0)
    res = sorted(res[-NUM_CLUSTERS:], key=lambda i: distances[i], reverse=True)
    topk = res[-NUM_CLUSTERS:]

    if topk[0] < num_clusters:
        return topk[0]
    return 0

def main():
    if len(sys.argv) != 3:
        raise ValueError("Usage: %s preamble num_clusters" % sys.argv[0])


    # Alternative (with file instead of FAISS)
    #centroids = numpy.loadtxt(CENTROIDS_FILE)
    #centroids = numpy.round(centroids * (1 << prec))

    components = numpy.load("/home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/pca_192.npy")

    tokenizer = AutoTokenizer.from_pretrained("/home/nsklab/yyh/similar/LeCaRDv2/Lawformer/Lawformer")
    model = AutoModel.from_pretrained("/home/nsklab/yyh/similar/LeCaRDv2/Lawformer/Lawformer")

    #end1 = time.time()
    #print("Setup: ", end1-start1)
    #print("  Ready to start embedding")

    for line in sys.stdin:
        line = line.rstrip()
        inputs = tokenizer(line, return_tensors="pt")
        outputs = model(**inputs)
        v = outputs.last_hidden_state[0, 0].tolist()
        v = numpy.array(v)
        v = numpy.round(v * (1 << prec)).astype('int') 

        #print("Find closest cluster: ", end3-end2)

        out = numpy.clip(numpy.round(numpy.matmul(v, components)/10), -16, 15).astype('int')
        sys.stdout.write(json.dumps({"Cluster_index": int(0), "Emb": out.tolist()}))
        sys.stdout.flush()
        #end4 = time.time()
        #print("PCA: ", end4-end3)

        #end = time.time()
        #print("Total: ", end-start2)

if __name__ == "__main__":
    main()
