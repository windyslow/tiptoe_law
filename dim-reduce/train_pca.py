from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample
from sklearn.preprocessing import normalize
import logging
import os
import gzip
import csv
import random
import numpy as np
import torch
import numpy
import sys
import glob
import re
import concurrent.futures

#New size for the embeddings
NEW_DIM = 192 
NUM_CLUSTERS = 1280
PCA_COMPONENTS_FILE = ("/home/nsklab/yyh/similar/tiptoe/search/mydata/data/1m1680_image_pca_%d.npy" % (NEW_DIM))

def train_pca(train_vecs):
    pca = PCA(n_components=NEW_DIM,svd_solver="full")
    pca.fit(train_vecs)
    return pca

def adjust_precision(vec):
    return numpy.round(numpy.array(vec) * (1<<6))

train_embeddings = numpy.load("/home/nsklab/yyh/similar/CM-project/query_case.npy",allow_pickle = True)
#train_embeddings = numpy.load("/work/edauterman/private-search/code/embedding/web_msmarco_reduce/web-idx-0.npy")
train_embeddings = [adjust_precision(embed) for embed in train_embeddings]
print("Loaded and adjusted precision")
pca = train_pca(train_embeddings)
print("Ran PCA")
numpy.save(PCA_COMPONENTS_FILE, numpy.transpose(pca.components_))
