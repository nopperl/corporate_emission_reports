# Extracting Greenhouse Gas Emission Values from Corporate Sustainability Reports

  * Web demo: https://huggingface.co/spaces/nopperl/emission-extractor
  * Evaluation dataset: https://huggingface.co/datasets/nopperl/corporate-emission-reports
  * Finetuning dataset: https://huggingface.co/datasets/nopperl/sustainability-report-emissions
    * Instruction-style JSONL: https://huggingface.co/datasets/nopperl/sustainability-report-emissions-instruction-style
  * Finetuned model: https://huggingface.co/nopperl/emissions-extraction-lora

Experiments on training and evaluating language models on the long-context structured information extraction task of extracting greenhouse gas emissions from corporate sustainability reports.

## Setup

Note: the setup assumes a cuda-compatible GPU and driver.

Clone the repository:

    git clone --recursive https://github.com/nopperl/corporate_emission_reports

Install the environment using [conda](https://docs.conda.io/en/latest/):

    conda env create --file environment.yaml
    conda activate emissions
    pip install -e .
    git lfs install

The system uses the [llama.cpp](https://github.com/ggerganov/llama.cpp) `main` binary directly to utilize [self-extend](https://github.com/datamllab/LongLM). For this, llama.cpp has to be built:

    cd llama.cpp
    mkdir build
    cd build
    cmake .. -DLLAMA_CUBLAS=1
    cmake --build . --config Release
    cd ..

### Models

Create a model directory and cd there. Default configuration (`model_config.json`) expects the directory to be at `./models`. If a different path is used, the configuration needs to be adapted.

```
mkdir models
cd models
```

At least one model needs to be downloaded to use the system.

#### Mistral

Download [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and adapt it for this system:

```
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
python ../llama.cpp/convert.py Mistral-7B-Instruct-v0.2 -outtype f16
curl https://huggingface.co/nopperl/Mistral-7B-Instruct-v0.2/raw/main/tokenizer_config.json --output Mistral-7B-Instruct-v0.2/tokenizer_config.json
```

##### Mistral LoRA

LoRA for Mistral-7B-Instruct-v0.2 finetuned on Mixtral outputs:

     git clone https://huggingface.co/nopperl/emissions-extraction-lora

#### OpenChat

Download and convert openchat-3.5-0106:

    git clone https://huggingface.co/openchat/openchat-3.5-0106
    python ../llama.cpp/convert.py openchat-3.5-0106 -outtype f16

#### Qwen-1.8B

Download and convert Qwen-1.8B-Chat:

    git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat
    python ../llama.cpp/convert-hf-to-gguf.py Qwen-1_8B-Chat --outtype f16

#### Mixtral

Download [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) and quantize it to Q5_K_M:

    git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
    python ../llama.cpp/convert.py Mixtral-8x7B-Instruct-v0.1 -outtype f16
    ../llama.cpp/build/bin/quantize Mixtral-8x7B-Instruct-v0.1/ggml-model-f16.gguf Q5_K_M
    
Note: if you run out of VRAM when running the scripts, consider a lower quantization type.

### Dataset (required for evaluation)

Return to the base directory and clone the dataset to the `data` directory.

    git clone https://huggingface.co/datasets/nopperl/corporate-emission-reports data

Retrieve the sustainability reports:

    python download_documents.py data/corp_emissions.parquet

Important: if the script fails to download the file for a report uid, download the file manually and put it into the `pdfs` directory with the following file format: `{uid}.pdf`. Afterwards, rerun the script to check if the file hash matches the original.

## Usage

`inference.py` can be used to run an existing model on a sustainability report. An example for the [emissions-extraction-lora](https://huggingface.co/nopperl/emissions-extraction-lora) on the [2022 sustainability report by the Bristol-Myers Squibb Company](https://www.bms.com/assets/bms/us/en-us/pdf/bmy-2022-esg-report.pdf):

    python inference.py --model mistral --lora models/emissions-extraction-lora/ggml-adapter-model.bin https://www.bms.com/assets/bms/us/en-us/pdf/bmy-2022-esg-report.pdf

Note: file paths can also be used as input. For example, using Mixtral and `pdfs/0066.pdf`:

    python inference.py --model mixtral pdfs/0088.pdf

### Evaluation

To run the models on the entire benchmark dataset:

```
python experiment.py --model qwen --max_group_neighbour_size 64 --max_group_window_size 2048
python experiment.py --model openchat --max_group_neighbour_size 16 --max_group_window_size 2048
python experiment.py --model mistral
python experiment.py --model mistral --lora models/emissions-extraction-lora/ggml-adapter-model.bin
python experiment.py --model mixtral
```

The outputs will be stored as JSON files in a model directory at `outputs`. To convert these into [Parquet](https://parquet.apache.org) files:

```
python outputs_to_parquet.py outputs/Qwen-1_8B-Chat/
python outputs_to_parquet.py outputs/openchat-3.5-0106/
python outputs_to_parquet.py outputs/Mistral-7B-Instruct-v0.2/
python outputs_to_parquet.py outputs/Mistral-7B-Instruct-v0.2-lora/
python outputs_to_parquet.py outputs/Mixtral-8x7B-Instruct-v0.1
```

Now, to evaluate the extracted emission values using strict, tolerant and graceful accuracy:

```
python evaluation.py --evaluation_type values
python evaluation.py --evaluation_type values --mode tolerant
python evaluation.py --evaluation_type values --mode graceful
```

To evaluate the cited source pages of the models:

    python evaluation.py --evaluation_type sources

To get statistics of the generated prompt length:

    python data_statistics.py prompts/generated/Mistral-7B-Instruct-v0.2/

## Finetune

### Reproduce training data (optional)

The training dataset consists of the model outputs of Mixtral on the sustainability reports in the [sustainability-report-emissions](https://huggingface.co/datasets/nopperl/sustainability-report-emissions) dataset. It can be reproduced in the following way: First, clone the source dataset and download all its sustainbility reports:

```
git clone https://huggingface.co/datasets/nopperl/sustainability-report-emissions
python download_documents.py sustainability-report-emissions pdfs_train
```

Next, to reproduce the [sustainability-report-emissions-instruction-style](https://huggingface.co/datasets/nopperl/sustainability-report-emissions-instruction-style) dataset using the model outputs in the [sustainability-report-emissions dataset](https://huggingface.co/datasets/nopperl/sustainability-report-emissions):

    python create_train_dataset.py --predictions_source emissions_train.parquet

The dataset will be at `data_train/emissions_sft.jsonl`.

Alternatively, the outputs can be reproduced as well using:

    python experiment.py --model mixtral --documents_dir pdfs_train --output_dir outputs_train --prompt_output_dir prompts/generated_train

Then, the dataset can be reproduced using the generated outputs:

    python create_train_dataset.py --predictions_source outputs_train/Mixtral-8x7B-Instruct-v0.1

### Training

The training is implemented using [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). For this, a different environment must be installed and activated:

    conda env create --file environment-axolotl.yaml
    conda activate axolotl
    pip install "axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl"

To finetune a LoRA for the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base model on the [sustainability-report-emissions-instruction-style](https://huggingface.co/datasets/nopperl/sustainability-report-emissions-instruction-style) dataset:

    accelerate launch -m axolotl.cli.train train_config/lora_sft.yml

It makes sense to adjust the training configuration based on the system. A different dataset (e.g. the `data_train/emissions_sft.jsonl` dataset reproduced above) can also be set.

The LoRA is stored in safetensors format at `emissions-extraction-lora` and can be converted into GGUF format consumable by llama.cpp:

    python llama.cpp/convert-lora-to-ggml.py emissions-extraction-lora

### Inference

Running inference with the trained LoRA:

    python inference.py test.pdf --model mistral --lora emissions-extraction-lora/ggml-adapter-model.bin
