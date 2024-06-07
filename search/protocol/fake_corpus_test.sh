#!/bin/bash

# Directory for intermediate files
mkdir -p interm

# Generate test corpus of 40,960 documents


# Test correctness of nearest-neighbor and url services
go test -timeout 0 -run Fake -medcorpus /home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/pca_embeddings_192.csv -preamble ../corpus
