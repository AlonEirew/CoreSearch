# Event-Search
An Information Retrieval system for corefering events

## Index Creation Scripts

### elastic_index.py
This script will create a new ElasticSearch index containing documents generated from the input file.
In case given index already exists, it will be deleted by this process and recreated.
```
Usage:
    elastic_index.py --input=<PassageFile> --index=<IndexName>

Options:
    -h --help                   Show this screen.
    --input=<PassageFile>       Passage input file to index into ElasticSearch
    --index=<IndexName>         The index name to create in ElasticSearch
```

### faiss_index.py

### training env params
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=true python src/index/faiss_index.py
