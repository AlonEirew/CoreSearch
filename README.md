# Event-Search
An Information Retrieval system for corefering events

## Index Creation Scripts

### elastic_index.py
This script will create a new ElasticSearch index containing documents generated from the input file.
In case given index already exists, it will be deleted by this process and recreated.
```
Prerequisite:
Pulling elastic image:
    docker pull docker.elastic.co/elasticsearch/elasticsearch:7.9.2
    
Running docker:
    docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2

Usage:
    elastic_index.py --input=<PassageFile> --index=<IndexName>

Options:
    -h --help                   Show this screen.
    --input=<PassageFile>       Passage input file to index into ElasticSearch
    --index=<IndexName>         The index name to create in ElasticSearch
```

### Running End-To-End
1) Generate DPR Files using script:
   `#>scripts/to_dpr_format.py`
2) Run **retriever** training:
   `#>src/train/train_retriever.py`
3) Run Evaluation and Index script on DEV to generate results and passage index:
   `#>src/evaluation/evaluate_retriever.py`
4) Take the best model and generate TRAIN and TEST index
5) Generate Squad files using script:
   `#>scripts/to_squad_format.py`
6) Run **reader** training
   `#>src/train/train_reader.py`
7) Run full pipeline on DEV set of retriever/reader
   `#>src/pipeline/run_haystack_pipeline.py`
8) Run full pipeline with best model on TEST set

### training env params
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=true python src/index/faiss_index.py
