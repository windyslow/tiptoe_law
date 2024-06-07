import sys
import time
import json
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy 

# New size for the embeddings
NUM_CLUSTERS = 1
new_dimension = 192
prec = 6

CENTROIDS_FILE = "../data/image_info/10_image_centroids.npy"
PCA_COMPONENTS_FILE = "%s/artifact/dim192/pca_192.npy"

preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor()
])

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
    centroids = numpy.loadtxt(CENTROIDS_FILE)
    #centroids = numpy.round(centroids * (1 << prec))

    components = numpy.load("../data/image_info/1m1680_image_pca_192.npy")

  
    #加载模型
    model = InceptionResnetV1(pretrained='vggface2').eval()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()


    for line in sys.stdin:
        line = line.rstrip()
        image_path = str(line)  # 替换为你的图片路径
        # 加载并预处理图片
        img = Image.open(image_path)
        img = preprocess(img)

        img = img.unsqueeze(0)
        with torch.no_grad():
            embedding = model(img)

        v = embedding.cpu().numpy()[0]
        result = find_nearest_clusters_from_file(centroids, [v], 10)
        v = numpy.round(v * (1 << prec)).astype('int') 
        #result = find_nearest_clusters_from_file(centroids, [v], 5)
        #print("Find closest cluster: ", end3-end2)

        out = numpy.clip(numpy.round(numpy.matmul(v, components)), -16, 15).astype('int')
        sys.stdout.write(json.dumps({"Cluster_index": int(result), "Emb": out.tolist()}))
        sys.stdout.flush()
        #end4 = time.time()
        #print("PCA: ", end4-end3)

        #end = time.time()
        #print("Total: ", end-start2)

if __name__ == "__main__":
    main()

