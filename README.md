# CoreSearch
This project is following our research paper: [Cross-document Event Coreference Search: Task, Dataset and Modeling](https://arxiv.org/abs/2210.12654)

## Content
1. [Dataset Files](#coresearch-dataset-files)
2. [Pre-trained Models](#coresearch-pre-trained-models)
3. [Models Training](#coresearch-models-training) 
   1. Project Installation
   2. Retriever Training
   3. Reader Training


## CoreSearch Dataset Files
Download CoreSearch dataset files from:</br> 
https://huggingface.co/datasets/biu-nlp/CoreSearch </br>

**OR** download the cleaner version **CoreSearchV2** dataset files from:</br> 
https://huggingface.co/datasets/biu-nlp/CoreSearchV2 </br>

Using the following code snippet will download the dataset to the cache folder:

```python
In [1]: from huggingface_hub import snapshot_download
In [2]: snapshot_download(repo_id="biu-nlp/CoreSearch", revision="main", repo_type="dataset")
```
* Save the output location of the downloaded CoreSearch snapshot folder

[//]: # (Then place them under the data/resources folder in the root directory </br>)
[//]: # (```bash)
[//]: # (mkdir --parents ./data/resources; mv -v data/tmp/datasets--biu-nlp--CoreSearch/snapshots/1aa21e8da4ab4804816ede85d88d3d5e62024401/* $_)
[//]: # (rm -rf data/tmp/)
[//]: # (```)

### CoreSearch Dataset folder structure:
- CoreSearch/**dpr**: Files in DPR format used for training the retriever
- CoreSearch/**squad**: Files in SQuAD format used for training the reader
- CoreSearch/**train**: The train used for generating the dpr/squad files
- CoreSearch/**clean**: The clean dataset files used for evaluation


## CoreSearch Pre-trained Models
Below links to models already pre-trained on the CoreSearch dataset. </br>
* Link to CoreSearch retriever model: [retriever](https://huggingface.co/biu-nlp/coresearch-retriever-spanbert)
* Link to CoreSearch reader model: [reader](https://huggingface.co/biu-nlp/coresearch-reader-roberta)


## CoreSearch Models Training
The instructions below explain how to train the retriever and reader models

### Project Installation
Installation from the source. Python's virtual or Conda environments are recommended.</br>
Project was tested with `Python 3.9` 

```bash
git clone https://github.com/AlonEirew/CoreSearch.git
cd CoreSearch
pip install -r requirements.txt
pip install -e .

# I set the path to the project in the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:/<replace_with_path>/CoreSearch"
```

### Retriever Training
Training the retriever moder require the CoreSearch data in DPR format (avilable in the dataset huggingface link above).
Full argument description is available in the top of `train_retriever.py` script. 
```bash
python src/train/train_retriever.py \
    --doc_dir [replace_with_hubs_cache_path]/dpr/ \
    --train_filename Train.json \
    --dev_filename Dev.json \
    --checkpoint_dir data/checkpoints/ \
    --output_model Retriever_SpanBERT \
    --add_special_tokens true \
    --n_epochs 5 \
    --max_seq_len_query 64 \
    --max_seq_len_passage 180 \
    --batch_size 64 \
    --query_model SpanBERT/spanbert-base-cased \
    --passage_model SpanBERT/spanbert-base-cased \
    --evaluate_every 500
```

### Reader Training
Training reader moder require the CoreSearch data in SQuAD format.
Full argument description is available in the top of `train_reader.py` script.
```bash
python src/train/train_reader.py \
    --doc_dir [replace_with_hubs_cache_path]/squad/ \ 
    --train_filename Train_squad_format_1pos_23neg.json \ 
    --dev_filename Dev_squad_format_1pos_23neg.json \ 
    --checkpoint_dir data/checkpoints/ \ 
    --output_model Reader-RoBERTa_base_Kenton \ 
    --predicting_head kenton \
    --num_processes 10 \ 
    --add_special_tokens true \ 
    --n_epochs 5 \
    --max_seq_len 256 \
    --max_seq_len_query 64 \
    --batch_size 24 \
    --reader_model roberta-base \
    --evaluate_every 750
```

## Models Evaluation
### Retriever Evaluation
This script is for evaluating the retriever model and generating a file index for the top-k results of every question.
Information on parameters can be found in the top of `evaluate_retriever.py` script.

```bash
python src/evaluation/evaluate_retriever.py \
    --query_filename [replace_with_hubs_cache_path]/train/Dev_queries.json \
    --passages_filename [replace_with_hubs_cache_path]/clean/Dev_all_passages.json \
    --gold_cluster_filename [replace_with_hubs_cache_path]/clean/Dev_gold_clusters.json \
    --query_model data/checkpoints/Retriever_SpanBERT_notoks_5it/0/query_encoder \
    --passage_model data/checkpoints/Retriever_SpanBERT_notoks_5it/0/passage_encoder \
    --out_index_file file_indexes/Dev_Retriever_spanbert_notoks_5it0_top500.json \
    --out_results_file file_indexes/Dev_Retriever_spanbert_notoks_5it0_top500_results.txt \
    --num_processes -1 \
    --add_special_tokens true \
    --max_seq_len_query 64 \
    --max_seq_len_passage 180 \
    --batch_size 240 \
    --top_k 500
```

### End2End Pipeline
**_Prerequisites:_** Generating an index to retrieve from, index can be generated by running `evaluate_retriever.py`, or by creating an elastic index using `elastic_index.py` (detailed below) for BM25 retriever.

Running the end2end pipeline.
Information on parameters can be found in the top of `run_e2e_pipeline.py` script.
```bash
python src/pipeline/run_e2e_pipeline.py \
  --predicting_head kenton \
  --max_seq_len_query 64 \
  --max_seq_len_passage 180 \
  --add_special_tokens true \
  --batch_size_retriever 24 \
  --batch_size_reader 24 \
  --top_k_retriever 500 \
  --top_k_reader 50 \
  --query_model data/checkpoints/Retriever_SpanBERT_5it/1/query_encoder \
  --passage_model data/checkpoints/Retriever_SpanBERT_5it/1/passage_encoder \
  --reader_model data/checkpoints/Reader-RoBERTa_base_Kenton_special/1 \
  --query_filename [replace_with_hubs_cache_path]/train/Dev_queries.json \
  --passages_filename [replace_with_hubs_cache_path]/clean/Dev_all_passages.json \
  --gold_cluster_filename [replace_with_hubs_cache_path]/clean/Dev_gold_clusters.json \
  --index_file file_indexes/Dev_Retriever_spanbert_5it1_top500.json \
  --out_results_file results/Dev_End2End_5it1.txt \
  --magnitude all
```

## Elastic Index for BM25 Retriever

### elastic_index.py
This script will create a new ElasticSearch index containing documents generated from the input file.
In case given index already exists, it will be deleted by this process and recreated.

**_Prerequisite:_** Pulling elastic image and running it:
```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.9.2
docker run -d -p 9200:9200 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.9.2
```

**After Index is up and running:**
```bash
python src/index/elastic_index.py \
  --input=data/resources/clean/Train_all_passages.json \
  --index=train
```

## Additional Scripts
### DPR Files Generation
Generate DPR Files from CoreSearch files:<br/>
`python scripts/to_dpr_format.py`

### SQuAD Files Generation
Generate SQuAD Files from CoreSearch files:<br/>
`python scripts/to_squad_format.py`

## Run Full Experiments Pipeline:
1) Run **retriever** training -- `src/train/train_retriever.py`
2) Run Evaluation and Index script on DEV to generate results and passage index: -- `src/evaluation/evaluate_retriever.py`
3) Take the best model and generate TRAIN and TEST index (using above `evaluate_retriever.py` script)
4) Generate Squad files using script -- `scripts/to_squad_format.py`
5) Run **reader** training -- `src/train/train_reader.py`
6) Run full pipeline on DEV set of retriever/reader -- `src/pipeline/run_e2e_pipeline.py`
7) Run full pipeline with best model on TEST set

[//]: # (## Run Demo)

[//]: # (1&#41; Running the Elasticsearch index: </br>)

[//]: # (`#>sudo docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2`)

[//]: # (2&#41; Create elasticsearch index &#40;using BM25 + ElasticSearch or Faiss&#41;)

[//]: # (`#>python src/index/elastic_wiki_index.py --input=../WikipediaToElastic-1.0/file_index --index=wiki`)

[//]: # (3&#41; Running the Rest API Server: </br>)

[//]: # (   * Login to the server and run the Rest API inside a screen instance: )

[//]: # (`#>CUDA_VISIBLE_DEVICES=0 gunicorn --access-logfile - rest_api.application:app -b 0.0.0.0:8081 -k uvicorn.workers.UvicornWorker -t 300`)

[//]: # (5&#41; Running the client: </br>)

[//]: # (   )
