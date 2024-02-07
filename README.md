# Extracting Greenhouse Gas Emission Values from Corporate Sustainability Reports

  * Web demo: https://huggingface.co/spaces/nopperl/emission-extractor
  * Evaluation dataset: https://huggingface.co/datasets/nopperl/corporate-emission-reports
  * Finetuning dataset: https://huggingface.co/datasets/nopperl/sustainability-report-emissions
    * Instruction-style JSONL: https://huggingface.co/datasets/nopperl/sustainability-report-emissions-instruction-style
    * Preferences JSONL (for DPO): https://huggingface.co/datasets/nopperl/sustainability-report-emissions-dpo
  * Finetuned model: https://huggingface.co/nopperl/emissions-extraction-lora

Experiments on training and evaluating language models on the long-context structured information extraction task of extracting greenhouse gas emissions from corporate sustainability reports. The finetuned [emissions-extraction-lora](https://huggingface.co/nopperl/emissions-extraction-lora) 7B model reaches an emission value extraction accuracy of 65\% (up from 46\% of the base model) and a source citation accuracy of 69\% (base model: 52\%) on the [corporate-emission-reports](https://huggingface.co/datasets/nopperl/corporate-emission-reports) dataset, matching the performance of the 45B [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).

## Setup

Note: for now, the setup assumes a CUDA-compatible GPU and driver. It should also work on CPU-only machines. Other configurations are not tested.

Clone the repository:

    git clone --recursive https://github.com/nopperl/corporate_emission_reports

Install the environment using [conda](https://docs.conda.io/en/latest/):

    conda env create --file environment.yaml
    conda activate emissions
    pip install -e .
    git lfs install

That is all if the [transformers](https://github.com/huggingface/transformers) inference engine is used. If the [llama.cpp](https://github.com/ggerganov/llama.cpp) should be used, it has to be built:

    cd llama.cpp
    mkdir build
    cd build
    cmake .. -DLLAMA_CUBLAS=1
    cmake --build . --config Release
    cd ..

(remove `LLAMA_CUBLAS` if no CUDA-compatible GPU is available.)

### Models

If llama.cpp is used, the models need to be downloaded before using the system. For this, create a model directory and cd there. Default configuration (`model_config.json`) expects the directory to be at `./models`. If a different path is used, the configuration needs to be adapted.

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
```

##### Mistral LoRA

Download the LoRA for Mistral-7B-Instruct-v0.2 finetuned on Mixtral outputs:

     git clone https://huggingface.co/nopperl/emissions-extraction-lora

#### OpenChat

Download and convert [openchat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106):

    git clone https://huggingface.co/openchat/openchat-3.5-0106
    python ../llama.cpp/convert.py openchat-3.5-0106 -outtype f16

#### Qwen-1.8B

Download and convert [Qwen-1.8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat):

    git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat
    python ../llama.cpp/convert-hf-to-gguf.py Qwen-1_8B-Chat --outtype f16

#### Mixtral

Download [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) and quantize it to Q5_K_M:

    git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
    python ../llama.cpp/convert.py Mixtral-8x7B-Instruct-v0.1 -outtype f16
    ../llama.cpp/build/bin/quantize Mixtral-8x7B-Instruct-v0.1/ggml-model-f16.gguf Q5_K_M
    mv Mixtral-8x7B-Instruct-v0.1/ggml-model-Q5_K_M.gguf Mixtral-8x7B-Instruct-v0.1/ggml-model-f16.gguf 
    curl https://huggingface.co/nopperl/Mistral-7B-Instruct-v0.2/raw/main/tokenizer_config.json --output Mixtral-8x7B-Instruct-v0.1/tokenizer_config.json
    
Note: if you run out of VRAM when running the scripts, consider a lower quantization type.

### Dataset (required for evaluation)

Return to the base directory and clone the dataset to the `data` directory.

    git clone https://huggingface.co/datasets/nopperl/corporate-emission-reports data

Retrieve the sustainability reports:

    python -m corporate_emission_reports.download_documents.py data/corp_emissions.parquet

Important: if the script fails to download the file for a report uid, download the file manually and put it into the `pdfs` directory with the following file format: `{uid}.pdf`. Afterwards, rerun the script to check if the file hash matches the original.

## Usage

`corporate_emission_reports/inference.py` can be used to run an existing model on a sustainability report. An example for the [emissions-extraction-lora](https://huggingface.co/nopperl/emissions-extraction-lora) on the [2022 sustainability report by the Bristol-Myers Squibb Company](https://www.bms.com/assets/bms/us/en-us/pdf/bmy-2022-esg-report.pdf) using [llama.cpp](https://github.com/ggerganov/llama.cpp) as inference engine:

    python -m corporate_emission_reports.inference --model mistral --lora models/emissions-extraction-lora/ggml-adapter-model.bin https://www.bms.com/assets/bms/us/en-us/pdf/bmy-2022-esg-report.pdf

Note: file paths can also be used as input. For example, using Mixtral and `pdfs/0066.pdf`:

    python -m corporate_emission_reports.inference --model mixtral pdfs/0088.pdf

For more inference usage examples (using [transformers](https://github.com/huggingface/transformers), etc.), refer to the [emissions-extraction-lora readme](https://huggingface.co/nopperl/emissions-extraction-lora#example-usage).

### Evaluation

To run the models on the entire benchmark dataset:

```
python -m corporate_emission_reports.experiment --model qwen --max_group_neighbour_size 64 --max_group_window_size 2048
python -m corporate_emission_reports.experiment --model openchat --max_group_neighbour_size 16 --max_group_window_size 2048
python -m corporate_emission_reports.experiment --model mistral
python -m corporate_emission_reports.experiment --model mistral --lora models/emissions-extraction-lora/ggml-adapter-model.bin
python -m corporate_emission_reports.experiment --model mixtral
```

The outputs will be stored as JSON files in a model directory at `outputs`. To convert these into [Parquet](https://parquet.apache.org) files:

```
python -m corporate_emission_reports.outputs_to_parquet outputs/Qwen-1_8B-Chat/
python -m corporate_emission_reports.outputs_to_parquet outputs/openchat-3.5-0106/
python -m corporate_emission_reports.outputs_to_parquet outputs/Mistral-7B-Instruct-v0.2/
python -m corporate_emission_reports.outputs_to_parquet outputs/Mistral-7B-Instruct-v0.2-lora/
python -m corporate_emission_reports.outputs_to_parquet outputs/Mixtral-8x7B-Instruct-v0.1
```

Now, to evaluate the extracted emission values using strict, tolerant and graceful accuracy:

```
python -m corporate_emission_reports.evaluation --evaluation_type values
python -m corporate_emission_reports.evaluation --evaluation_type values --mode tolerant
python -m corporate_emission_reports.evaluation --evaluation_type values --mode graceful
```

To evaluate the cited source pages of the models:

    python -m corporate_emission_reports.evaluation --evaluation_type sources

To get statistics of the generated prompt length:

    python -m corporate_emission_reports.data_statistics prompts/generated/Mistral-7B-Instruct-v0.2/

## Finetune

### Reproduce training data (optional)

The training dataset consists of the model outputs of Mixtral on the sustainability reports in the [sustainability-report-emissions](https://huggingface.co/datasets/nopperl/sustainability-report-emissions) dataset. It can be reproduced in the following way: First, clone the source dataset and download all its sustainbility reports:

```
git clone https://huggingface.co/datasets/nopperl/sustainability-report-emissions
python -m corporate_emission_reports.download_documents sustainability-report-emissions/emissions_train.parquet pdfs_train
```

Next, to reproduce the [sustainability-report-emissions-instruction-style](https://huggingface.co/datasets/nopperl/sustainability-report-emissions-instruction-style) dataset using the model outputs in the [sustainability-report-emissions dataset](https://huggingface.co/datasets/nopperl/sustainability-report-emissions):

    python -m corporate_emission_reports.create_train_dataset --predictions_source sustainability-report-emissions/emissions_train.parquet

The dataset will be at `data_train/emissions_sft.jsonl`.

To reproduce the [sustainability-report-emissions-dpo](https://huggingface.co/datasets/nopperl/sustainability-report-emissions-dpo) dataset:

    python -m corporate_emission_reports.create_train_dataset --predictions_source sustainability-report-emissions/emissions_train.parquet --type dpo --dataset_path data_train/emissions_dpo.jsonl 

Alternatively, the outputs can be reproduced as well using:

    python -m corporate_emission_reports.experiment --model mixtral --documents_dir pdfs_train --output_dir outputs_train --prompt_output_dir prompts/generated_train

Then, the dataset can be reproduced using the generated outputs:

    python -m corporate_emission_reports.create_train_dataset --predictions_source outputs_train/Mixtral-8x7B-Instruct-v0.1

### Training

The training is implemented using [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). For this, a different environment must be installed and activated:

    conda env create --file environment-axolotl.yaml
    conda activate axolotl
    pip install "axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl"

To finetune a LoRA for the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base model on the [sustainability-report-emissions-instruction-style](https://huggingface.co/datasets/nopperl/sustainability-report-emissions-instruction-style) dataset:

    accelerate launch -m axolotl.cli.train train_config/lora_sft.yml

It makes sense to adjust the training configuration based on the system. A different dataset (e.g. the `data_train/emissions_sft.jsonl` dataset reproduced above) can also be set.

Instead, to finetune a LoRA for the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base model using DPO on the [sustainability-report-emissions-dpo](https://huggingface.co/datasets/nopperl/sustainability-report-emissions-dpo) dataset:

    accelerate launch -m axolotl.cli.train train_config/lora_dpo.yml

The LoRA is stored in safetensors format at `emissions-extraction-lora` and can be converted into GGUF format consumable by llama.cpp:

    python llama.cpp/convert-lora-to-ggml.py emissions-extraction-lora

Furthermore, the adapter can be merged into the base model and converted into GGUF format:

    python -m corporate_emission_reports.merge_base_lora emissions-extraction-lora emissions-extraction-lora-merged-GGUF
    python llama.cpp/convert.py emissions-extraction-lora-merged-GGUF --outtype f16
    llama.cpp/build/bin/quantize emissions-extraction-lora-merged-GGUF/ggml-model-f16.gguf Q5_K_M

### Inference

Running inference with the trained LoRA:

    python -m corporate_emission_reports.inference test.pdf --model mistral --lora emissions-extraction-lora/ggml-adapter-model.bin
