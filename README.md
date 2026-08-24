# FedMbio

## Content

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Installation environment](#installation-environment)
- [Model training and testing](#model-training-and-testing)
  - [a. Cohort-wise 5-fold split](#a-cohort-wise-5-fold-split)
  - [b. Centralized 5-fold split](#b-centralized-5-fold-split)
  - [c. FedMbio model training](#c-FedMbio-model-training)
  - [d. Benchmark: local training per client](#d-Benchmark-local-training-per-client)
  - [e. Benchmark: centralized setting](#e-Benchmark-centralized-setting)
- [Example](#Example)   

## Introduction

FedMbio, a privacy-preserving federated learning framework that enables collaborative, cross-institutional microbiome-based disease detection.

## Package requirement

* env.yml

## Installation environment

```
1.git clone https://github.com/qdu-bioinfo/FedMbio.git
2.cd FedMbio
3.conda env create -f env.yml
4.conda activate FedMbio
```

###  	Download DistilBERT model files (first-time setup)

* On the **first run**, this project requires the pretrained **DistilBERT** model files:
   `distilbert/distilbert-base-uncased`

* Download link (Hugging Face):
    https://huggingface.co/distilbert/distilbert-base-uncased

* Please download the model from Hugging Face and place it into the following directory:↳

  ```
  FedMbio/distilbert-base-uncased/
  ```

  After downloading, the folder should contain files such as:

  * `config.json`
  * `model.safetensors`
  * `vocab.txt`
  * `tokenizer.json` 
  * `tokenizer_config.json`

  > **Important:** Keep the folder name exactly as `distilbert-base-uncased`, since the code loads the model from this local path.


## Model training and testing

### a. Cohort-wise 5-fold split
* Preserve cohort boundaries and avoid cross-cohort leakage. Each cohort is split **independently** into 5 folds.

```
python GeneratorData.py --data_type wgs --groups CTR_ADA
```
### b. Centralized 5-fold split

* Build a centralized baseline by first **merging all cohorts** into a single dataset and then performing a global 5-fold split.	

```
python GeneratorData_Central.py --data_type wgs --groups CTR_ADA
```

### c. FedMbio Model Training and Evaluation

```
python FedMbio.py --data_type wgs --groups CTR_ADA --num_clients 9
```

### d. Benchmark: Local Training and Evaluation per Institution

```
python ./BaselineDir/BaseLine.py --data_type wgs --groups CTR_ADA --num_clients 9
```
### e. Benchmark: Centralized Training and Evaluation

```
python ./BaselineDir/BaseLine_Central.py --data_type wgs --groups CTR_ADA
```
## Example

### a. Running CTR vs. CRC analysis on WGS sequencing data.

```
python TestFedMbio.py --data_type wgs --groups CTR_CRC --num_clients 12 --model_dir "Models/wgs/CTRCRC"
```

### b. Running CTR vs. ADA analysis on WGS sequencing data.

```
python TestFedMbio.py --data_type wgs --groups CTR_ADA --num_clients 9 --model_dir "Models/wgs/CTRADA"
```
### c. Running CTR vs. CRC analysis on 16S sequencing data.

```
python TestFedMbio.py --data_type 16s --groups CTR_CRC --num_clients 6 --model_dir "Models/16s/CTRCRC"
```
### d. Running CTR vs. ADA analysis on 16S sequencing data.

```
python TestFedMbio.py --data_type 16s --groups CTR_ADA --num_clients 5 --model_dir "Models/16s/CTRADA"
```

