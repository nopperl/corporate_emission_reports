# Extracting Greenhouse Gas Emission Values from Corporate Sustainability Reports

  * Web demo: https://huggingface.co/spaces/nopperl/emission-extractor
  * Evaluation dataset: https://huggingface.co/datasets/nopperl/corporate-emission-reports
  * Finetuning dataset: https://huggingface.co/datasets/nopperl/sustainability-report-emissions
    * Instruction-style JSONL: https://huggingface.co/datasets/nopperl/sustainability-report-emissions-instruction-style
    * Preferences JSONL (for DPO): https://huggingface.co/datasets/nopperl/sustainability-report-emissions-dpo
  * Finetuned model: https://huggingface.co/nopperl/emissions-extraction-lora

Experiments on training and evaluating language models on the long-context structured information extraction task of extracting greenhouse gas emissions from corporate sustainability reports. The finetuned [emissions-extraction-lora](https://huggingface.co/nopperl/emissions-extraction-lora) 7B model reaches an emission value extraction accuracy of 65\% (up from 46\% of the base model) and a source citation accuracy of 77\% (base model: 53\%) on the [corporate-emission-reports](https://huggingface.co/datasets/nopperl/corporate-emission-reports) dataset, matching the performance of the 45B [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).

## Table of Contents
1. [Overview](#overview)
2. [Results](#results)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Future](#future)


## Overview

### Motivation
Data about corporate greenhouse gas emissions is usually published only as part of sustainability report PDF's, which is not a machine-readable format. Interested actors have to manually extract emission data from these reports, which is a tedious and time-consuming process. An automatic information-extraction system could solve this issue.

### Contributions

  * A manually-created evaluation dataset (N=100),
  * a synthetic finetuning dataset (N=3233),
  * an evaluation of multiple models (1.8B-45B) covering prompt strategies, numerical issues and self extend,
  * an evaluation of different finetuning configurations,
  * a [finetuned Mistral model](https://huggingface.co/nopperl/emissions-extraction-lora) which nearly matches and partially exceeds Mixtral on this task, and
  * a web demo.

### Description
This project benchmarks and finetunes large language models for the task mentioned in the motivation. Since sustainability reports are quite long on average, extracting information from them is a requires a long context. Furthermore, since the output needs to be able to be parsed by machines, the models need to output structured information. Hence, this project also provides indications on how language models can be used for long-context structured information extraction tasks in general.

For this purpose, two datasets are created:
  * an evaluation dataset of 100 sustainability reports from geographically-diverse corporations and manually-extracted emission values, and
  * a finetuning dataset of 3233 different geographically-diverse sustainability reports and emission values extracted by [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).

The former dataset is used to benchmark the models listed in the below table. The focus is on <=7B models as they require less resources. [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) is used as SotA model of this class. To evaluate how a similar model with lower context size performs when using [self-extend](https://github.com/datamllab/LongLM), [openchat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106) is also tested (7B models with even lower context size of 2048 did not perform well enough in preliminary experiments). Furthermore, to investigate how large the parameter size needs to be for this task, the significantly smaller Qwen-1.8B-Chat model is also evaluated (similar models such as [phi-2](https://huggingface.co/microsoft/phi-2) or [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) did not produce useful output in preliminary experiments). To ascertain the upper limit, the significantly larger Mixtral is also evaluated. Mixtral is used as it performed significantly better than other >45B models (Llama-2-70B, Qwen-72B, goliath, deepseek-llm-67b) in preliminary experiments (a full evaluation of larger models was impossible due to resource constrains). Unsurprisingly, Mixtral performs significantly better than the others models under evaluation. To investigate whether this gap can be closed, the best performing "small" model Mistral is finetuned (using LoRA) on the latter dataset of Mixtral outputs on different sustainability reports.

model | param size | context length
--- | --- | ---
[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | 7B | 32768
[openchat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106) | 7B | 8192
[Qwen-1.8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat) | 1.8B | 8192
[Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 45B | 32768


The prompt consists of a carefully crafted instruction (see `corporate_emission_reports/prompt-templates/simple.txt`) and semi-structured XHTML text extracted from sustainability reports using [PyMuPDF](https://pymupdf.readthedocs.io/). Since the reports are long, a RAG-like system is used to insert the extracted text. First, the report is split into semantical chunks, in this case pages (i.e. each page is a chunk). Then, each page that contains relevant terms such as `scope 1` is added to the prompt. This setup is simple but fast and performs well in practice.

| | max | mean | median | min |
| --- | --- | --- | --- | --- |
| prompt token length | 60063 | 14544 | 12184 | 1004 |

The above table shows the token length distribution of the resulting prompts. As they are still quite long , the recent [self-extend](https://github.com/datamllab/LongLM) technique is used for post-training context-length extension.

The models are instructed in the prompt to output scope 1, 2 and 3 greenhouse gas emission values in metric tons of CO2eq as well as the chunks (i.e., pages) used for extraction. The latter is useful for manual verification if accurate enough. The instruction prompts for an output in a JSON schema defined by the Pydantic model in `corporate_emission_reports/pydantic_types.py`. To ensure this output, [BNF grammar-based decoding] and [lm-format-enforcer] are used.

For the evaluation, [llama.cpp](https://github.com/ggerganov/llama.cpp) is used as inference engine as it performed the fastest and required the least memory on the relatively old system used for experiments. Furthermore, its `main` binary already implements [self-extend](https://github.com/datamllab/LongLM).

For model finetuning, ZeRO Stage 3, Flash Attention 2, bfloat16 and the bitsandbytes AdamW optimizer are used to fit the long training sequences and increase utilization. Without these, only sequences up to a length of 6144 could be trained. Prompt construction and tokenization is kept consistent between finetuning and inference.

## Results

Performance is measured using two metrics:

1. accuracy for every extracted emission value
2. source page retrieval accuracy

scope 1 | scope 2 | scope 3 | avg of scopes | sources | model
--- | --- | --- | --- | --- | ---
49 | 34 | 54 | _46_ | 53 | mistral
33 | 31 | 56 | _40_ | 48 | openchat
12 | 8 | 5 | _8_ | 3 | qwen-1.8B
70 | 71 | 69 | _69_ | 64 | mixtral
65 | 62 | 69 | _65_ | 77 | lora

The above table unsurprisingly shows that Mixtral significantly outperforms the smaller models. Out of the smaller models, Mistral performs the best. Qwen-1.8B-Chat does not yield good results, but it is still surprising that it is even able to produce relevant outputs for emission values (which similar models such as phi-2](https://huggingface.co/microsoft/phi-2) or [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) failed to do). The [emission-extraction-lora](https://huggingface.co/nopperl/emissions-extraction-lora) model finetuned on Mixtral outputs remarkably nearly matches or partially exceeds Mixtral, which is important, as a 7B model is far easier to deploy than a 45B model.

It is notable that scope 2 accuracy is significantly lower for all models except Mixtral, likely due to oftentimes both market-based and location-based scope 2 emissions being present.

### Numerical issues

Since language models struggle with numerical tasks, it is investigated whether wrong extractions are due to numerical errors instead of failures in logic. The below table shows the performance with accuracy measured by interpreting values within 10% of each other as matching. All models significantly perform better using this metric, confirming that they stuggle with reproducing numbers correctly. Notably, this is also true for Mixtral.

scope 1 | scope 2 | scope 3 | avg of scopes | model
--- | --- | --- | --- | --- 
60 | 39 | 61 | _53_ |  mistral
43 | 43 | 60 | _49_ | openchat
14 | 12 | 5 | _10_ | qwen-1.8B
78 | 75 | 71 | _75_ | mixtral
70 | 66 | 71 | _69_ | lora

Furthermore, it is possible that extraction failures are due to wrong conversions of numbers into metric tons. This is investigated by computing accuracy irrespective of the unit. Again, the results in the below table show that the models perform better using this metric, confirming that they struggled with converting the numbers in the report to the correct unit.

scope 1 | scope 2 | scope 3 | avg of scopes | model
--- | --- | --- | --- | --- 
54 | 36 | 59 | _50_ |  mistral
35 | 35 | 62 | _44_ | openchat
17 | 11 | 8 | _12_ | qwen-1.8B
76 | 83 | 76 | _78_ | mixtral
79 | 78 | 81 | _79_ | lora

### Prompt engineering

scope 1 | scope 2 | scope 3 | avg of scopes | sources | model | type
--- | --- | --- | --- | --- | --- | ---
38 | 31 | 44 | _38_ | 52 | mistral | no starting answer
49 | 34 | 54 | _46_ | 53 | mistral | starting answer

Engineering good prompts is important to improve model performance. The above table compares default ChatML prompts and ChatML prompts with the assistant's answer already started by: ` I have extracted the Scope 1, 2 and 3 emission values from the document, converted them into metric tons and put them into the following json object:\n```json\n`. The results clearly show that the latter prompt leads to a better performance. Interestingly, it does not affect the performance for source citation. As this is not mentioned in the starting answer, this could be interpreted as additional indication of the starting answer's effect.

### Self-extend

scope 1 | scope 2 | scope 3 | avg of scopes | sources | failures | model | self-extend
--- | --- | --- | --- | --- | --- | --- | ---
40 | 32 | 50 | _41_ | 30 | 40 | openchat | no
33 | 31 | 56 | _40_ | 48 | 4 | openchat | yes

The openchat model is used to evaluate the usefulness of [self-extend](https://github.com/datamllab/LongLM). The above table shows that running the openchat model without self-extend failed for 40 out of 100 reports, while it failed only on 4 reports when using self-extend. These crashes occur for long input prompts. This strongly shows that self-extend is able to significantly extend the context size of the model. Furthermore, it does not decrease performance, although the metric distribution changes in favour of scope 3 emissions. Interestingly, it significantly increases the source accuracy.

### Finetuning

scope 1 | scope 2 | scope 3 | avg of scopes | sources | learning rate | epochs | type
--- | --- | --- | --- | --- | --- | --- | ---
49 | 34 | 54 | _46_ | 53 | - | - | mistral
70 | 71 | 69 | _69_ | 64 | - | - | mixtral
65 | 62 | 69 | _65_ | 77 | 2e-5 | 3 | lora

The Mistral-7B model is finetuned by training a LoRA on a dataset extracted by Mixtral for 3 epochs. The finetuned model not only significantly outperforms the base model, but remarkably nearly matches and partially exceeds the performance of Mixtral, at a much smaller size.

#### DPO

scope 1 | scope 2 | scope 3 | avg of scopes | sources | learning rate | epochs | type
--- | --- | --- | --- | --- | --- | --- | ---
49 | 34 | 54 | _46_ | 53 | - | - | base
43 | 37 | 55 | _45_ | 60 | 5e-6 | 1 | sft
32 | 32 | 49 | _38_ | 58 | 5e-6 | 1 | dpo

The LoRA was also trained using DPO by constructing a binary preferences dataset with randomly generated rejected outputs. The above table shows that the model finetuned using DPO performs significantly worse than the one using supervised finetuning.

#### ranks

scope 1 | scope 2 | scope 3 | avg of scopes | sources | learning rate | epochs | rank
--- | --- | --- | --- | --- | --- | --- | ---
38 | 35 | 47 | _40_ | 59 | 5e-6 | 1 | 4
43 | 37 | 55 | _45_ | 60 | 5e-6 | 1 | 32
42 | 37 | 54 | _44_ | 59 | 5e-6 | 1 | 64
56 | 38 | 57 | _47_ | 56 | 5e-6 | 1 | 128

An important hyperparameter for training LoRA's is the LoRA rank. The above table shows that a rank of 128 performed the best, with 32 performing the second-best. However, this is not as clear anymore when a different learning rate is used, as can be seen in the table below. Hence, since a rank of 32 performs at or better than 128 and is smaller, this value was used for all other training runs.

scope 1 | scope 2 | scope 3 | avg of scopes | sources | learning rate | epochs | rank
--- | --- | --- | --- | --- | --- | --- | ---
64 | 66 | 74 | _68_ | 65 | 2e-5 | 1 | 32
60 | 58 | 69 | _62_ | 69 | 2e-5 | 1 | 128

#### learning rate

As mentioned above, a learning rate of 2e-5 led to better results than 5e-6 both at epoch 1 and 3, as can be seen below.

scope 1 | scope 2 | scope 3 | avg of scopes | sources | learning rate | epochs | rank
--- | --- | --- | --- | --- | --- | --- | ---
43 | 37 | 55 | _45_ | 60 | 5e-6 | 1 | 32
64 | 66 | 74 | _68_ | 65 | 2e-5 | 1 | 32
60 | 57 | 68 | _62_ | 67 | 5e-6 | 3 | 32
65 | 62 | 69 | _65_ | 77 | 2e-5 | 3 | 32

#### epochs

It can be seen above that performance increases significantly from epoch 1 to 3 for a learning rate of 5e-6. The below table shows the case for a learning rate of 2e-5. Interestingly, scope 2 and 3 accuracy decreases after the first epoch, while source accuracy increases significantly. At epoch 4, all metrics except scope 1 accuracy decrease. On average, the performance at 3 epochs is slightly better, therefore this model is used.

scope 1 | scope 2 | scope 3 | avg of scopes | sources | learning rate | epochs
--- | --- | --- | --- | --- | --- | ---
64 | 66 | 74 | _68_ | 65 | 2e-5 | 1
65 | 62 | 69 | _65_ | 77 | 2e-5 | 3
67 | 59 | 68 | _65_ | 69 | 2e-5 | 4

#### Qwen-1.8B

As the Qwen-1.8B model is even easier to deploy than Mistral-7B, it was also investigated whether finetuning it increases its performance to a sufficient level. [Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) was used as base model due to its better support. The below table shows that the performance of the finetuned model is better, but still far from sufficient.

scope 1 | scope 2 | scope 3 | avg of scopes | sources | learning rate | epochs | type
--- | --- | --- | --- | --- | --- | --- | ---
12 | 8 | 5 | _8_ | 3 |  - | - | qwen-1.8B
3 | 5 | 22 | _10_ | 6 | 2e-4 | 4 | qwen-1.8B lora

### Additional findings

For some reason, the prompt processing batch size affects the output quality when using llama.cpp (see https://github.com/ggerganov/llama.cpp/issues/249), with quality being loosely negatively correlated to batch size. In this project, a batch size of 32 yielded the best trade-off between output quality and processing time.

Interestingly, different self-extend hyperparameters yielded the optimal performance for different models. The below table lists the used values. The neighbour size especially is unusually and counterintuitively high.

model | neighbour size | group size
--- | --- | ---
mistral | 8 | 1024
mixtral | 8 | 1024
openchat | 16 | 2048
qwen | 64 | 2048


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

(Remove `LLAMA_CUBLAS` if no CUDA-compatible GPU is available.)

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

To finetune [Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) instead of [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2):

    accelerate launch -m axolotl.cli.train train_config/lora_sft_qwen.yml

The LoRA is stored in safetensors format at `emissions-extraction-lora` and can be converted into GGUF format consumable by llama.cpp:

    python llama.cpp/convert-lora-to-ggml.py emissions-extraction-lora

Furthermore, the adapter can be merged into the base model and converted into GGUF format:

    python -m corporate_emission_reports.merge_base_lora emissions-extraction-lora emissions-extraction-lora-merged-GGUF
    python llama.cpp/convert.py emissions-extraction-lora-merged-GGUF --outtype f16
    llama.cpp/build/bin/quantize emissions-extraction-lora-merged-GGUF/ggml-model-f16.gguf Q5_K_M

### Inference

Running inference with the trained LoRA:

    python -m corporate_emission_reports.inference test.pdf --model mistral --lora emissions-extraction-lora/ggml-adapter-model.bin


## Future

Since PDF documents also contain visual information, it would be interesting to evaluate (and possibly finetune) multimodal vision-language models such as CogVLM.

The prompts could be improved using strategies such as few-shot or chain-of-thought prompting. However, these require a lot of tokens, which is problematic given the already high amount of tokens in input prompts.

While the focus was on large decoder-only language models, it may be interesting to test how smaller encoder(-decoder) models such as (XLM-)RoBERTa or DeBERTa perform. These models could be finetuned on the Mixtral output dataset. However, the average input sequence is far longer than their context size.

