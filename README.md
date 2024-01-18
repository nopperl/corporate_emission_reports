# Extracting Greenhouse Gas Emission Values from Corporate Sustainability Reports

  * Dataset: https://huggingface.co/datasets/nopperl/corporate-emission-reports

A manually-collected dataset and system to benchmark language models on the long-context information extraction task of extracting greenhouse gas emissions from corporate sustainability reports.

The system uses the [llama.cpp](https://github.com/ggerganov/llama.cpp) `main` binary directly to utilize [self-extend](https://github.com/datamllab/LongLM).

## Setup

Note: the setup assumes a cuda-compatible GPU and driver.

Clone the repository:

    git clone --recursive https://github.com/nopperl/corporate_emission_reports


Install the environment using [conda](https://docs.conda.io/en/latest/):

```
conda env create --file environment.yaml
conda activate emissions
pip install -r requirements.txt
```

Build [llama.cpp](https://github.com/ggerganov/llama.cpp):

```
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=1
cmake --build . --config Release
cd ..
```

### Models

Create a model directory and cd there.

```
mkdir models
cd models
```

#### Mistral

Download Mistral:

    git clone https://huggingface.co/nopperl/Mistral-7B-Instruct-v0.2

Alternatively, download it from original sources and adapt it for this system:

```
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
python ../llama.cpp/convert.py Mistral-7B-Instruct-v0.2 -outtype f16
curl https://huggingface.co/nopperl/Mistral-7B-Instruct-v0.2/raw/main/tokenizer_config.json --output Mistral-7B-Instruct-v0.2/tokenizer_config.json
```


Download and convert openchat-3.5-0106:

    git clone https://huggingface.co/openchat/openchat-3.5-0106
    python ../llama.cpp/convert.py openchat-3.5-0106 -outtype f16

Download and convert Qwen-1.8B-Chat:

    git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat
    python ../llama.cpp/convert-hf-to-gguf.py Qwen-1_8B-Chat --outtype f16

Note: if you run out of VRAM when using the system, consider quantizing the GGUF model.

### Dataset

Return to the base directory and clone the dataset to the `data` directory.

    git clone https://huggingface.co/datasets/nopperl/corporate-emission-reports data

Retrieve the sustainability reports:

    python download_documents.py data/corp_emissions.parquet

Important: if the script fails to download the file for a report uid, download the file manually and put it into the `pdfs` directory with the following file format `{uid}.pdf`. Afterwards, rerun the script to check if the file hash matches the original.


## Usage

To test an existing model on a sustainability report, in this example for `test.pdf`:

    python inference.py test.pdf --model mistral

Note: URLs can also be used as input.

To run the models on the entire benchmark dataset:

```
python experiment.py --model mistral
python experiment.py --model openchat --max_group_neighbour_size 16 --max_group_window_size 2048
python experiment.py --model qwen --max_group_neighbour_size 64 --max_group_window_size 2048
```

The outputs will be stored as JSON files in a model directory at `outputs`. To convert these into [Parquet](https://parquet.apache.org) files:

```
python outputs_to_parquet.py outputs/Mistral-7B-Instruct-v0.2/
python outputs_to_parquet.py outputs/openchat-3.5-0106/
python outputs_to_parquet.py outputs/Qwen-1_8B-Chat/
```

Now, to evaluate the extracte emission values using strict, tolerant and graceful accuracy:

```
python evaluation.py --evaluation_type values
python evaluation.py --evaluation_type values --mode tolerant
python evaluation.py --evaluation_type values --mode graceful
```

To evaluate the cited source pages of the models:

    python evaluation.py --evaluation_type sources

To get statistics of the generated prompt length:

    python data_statistics prompts/generated/Mistral-7B-Instruct-v0.2/
