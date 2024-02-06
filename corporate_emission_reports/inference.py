#!/usr/bin/env python
from argparse import ArgumentParser
from json import load, loads
from math import ceil
import os
from os.path import dirname, isfile, join
from tempfile import NamedTemporaryFile
from typing import List, Optional, Union
from subprocess import check_output
from urllib.request import urlretrieve
from urllib.parse import urlparse

from corporate_emission_reports.pydantic_types import Emissions, pydantic_model_to_grammar
from corporate_emission_reports.download_documents import download_document
from corporate_emission_reports.extraction import extract_chunks_from_document

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch


def try_download_document(document) -> Optional[str]:
     if urlparse(document).scheme not in ["https", "http", "ftp"]:
         return None
     return urlretrieve(document)[0]


def inference_llamacpp(prompt_token_len, prompt_file, model_path, eos, model_context_size=32768, grammar_file="", seed=123, max_group_neighbour_size=16, max_group_window_size=1024, lora=None):
    context_size = prompt_token_len + 120
    # Enable self-extend if the prompt does not fit into the models context size
    if context_size > model_context_size: 
        group_neighbour_size = max_group_neighbour_size
        group_window_size = min(int(model_context_size / 2), max_group_window_size)
    else:
        group_neighbour_size = 1  # 1 is disabled
        group_window_size = 512
    env_vars = os.environ.copy()
    env_vars.update({
        "MODEL_PATH": join(model_path, "ggml-model-f16.gguf"),
        "CONTEXT_SIZE": str(context_size),
        "PROMPT_FILE": prompt_file,
        "SEED": str(seed),
        "GRAMMAR_FILE": grammar_file,
        "GRP_ATTN_N": str(group_neighbour_size),
        "GRP_ATTN_W": str(group_window_size),
    })
    if lora:
        env_vars["LORA"] = lora
        script = "inference-llamacpp-lora.sh"
    else:
        script = "inference-llamacpp.sh"
    result = check_output(["sh", script], env=env_vars)
    output = result.decode()
    # Strip eos token if printed
    if output[-len(eos):] == eos:
        output = output[:-len(eos)]
    print("Output:", output)
    return Emissions.model_validate_json(output, strict=True)


def inference_hf(prompt_text, model_path, tokenizer, seed=123, max_group_neighbour_size=16, max_group_window_size=1024, lora=None):
    device_map = "auto" if torch.cuda.is_available() else None
    if not lora:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map=device_map)
    else:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(lora, trust_remote_code=True, low_cpu_mem_usage=True, device_map=device_map)
    prompt_tokenized = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    parser = JsonSchemaParser(Emissions.model_json_schema())
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    outputs = model.generate(prompt_tokenized, max_new_tokens=120, prefix_allowed_tokens_fn=prefix_function)
    output = outputs[0][prompt_tokenized.shape[1]:]
    if tokenizer.eos_token:
        output = output[:-1]
    output = tokenizer.decode(output)
    print("Output:", output)
    return Emissions.model_validate_json(output, strict=True)
    

def construct_prompt(document, tokenizer, prompt_template=None, extraction_mode="xhtml", return_tokenized=False):
    if not prompt_template:
        prompt_template = join(dirname(__file__), "prompt-templates/simple.txt")
    with open(prompt_template, "r") as prompt_file:
        instruction = prompt_file.read()
    user_message = extract_chunks_from_document(document, mode=extraction_mode)
    if not user_message:
        emissions = Emissions(scope_1=None, scope_2=None, scope_3=None, sources=[])
        return emissions
    prompt = [{"role": "user", "content": instruction + "\n" + user_message}]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    bos_len = 0 if tokenizer.bos_token is None else len(tokenizer.bos_token)
    prompt_text = prompt_text[bos_len:]  # strip bos, since both llama.cpp and hf automatically add this
    prompt_text += ' I have extracted the Scope 1, 2 and 3 emission values from the document, converted them into metric tons and put them into the following json object:\n```json\n'
    if return_tokenized:
        return prompt_text, tokenizer.encode(prompt_text)
    return prompt_text


def extract_emissions(document, model_path, prompt_template=None, model_context_size=32768, prompt_output_path=None, grammars_dir="grammars", extraction_mode="xhtml", seed=123, max_group_neighbour_size=8, max_group_window_size=2048, lora=None, engine="llama.cpp") -> Emissions:
    set_seed(seed)
    if not isfile(document):
        document = try_download_document(document)
        if document is None:
            raise ValueError("document needs to be a path to an existing file or an URL to a network object")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = construct_prompt(document, tokenizer, prompt_template, extraction_mode)

    if prompt_output_path:
        with open(prompt_output_path, "w") as prompt_file:
            prompt_file.write(prompt + "\n")

    if engine == "llama.cpp":
        grammar = pydantic_model_to_grammar(Emissions)
        grammar_path = f"{grammars_dir}/emissions_json.gbnf"
        with open(grammar_path, "w") as f:
            f.write(grammar)
        if not prompt_output_path:
            prompt_file = NamedTemporaryFile("w", delete=False)
            prompt_output_path = prompt_file.name
            with prompt_file:
                prompt_file.write(prompt + "\n")
        prompt_tokenized = tokenizer.encode(prompt)
        token_length = len(prompt_tokenized)
        eos_token = tokenizer.eos_token
        if "QWenTokenizer" in str(type(tokenizer)):
            token_length = int(token_length * 1.2)  # HF and llama.cpp use tokenizers that produce different lengths for QWen
            eos_token = "[PAD151643]"  # llama.cpp tokenizer uses this as eos
        emissions = inference_llamacpp(token_length, prompt_output_path, model_path, eos_token, model_context_size=model_context_size, grammar_file=grammar_path, seed=seed, max_group_neighbour_size=max_group_neighbour_size, max_group_window_size=max_group_window_size, lora=lora)
    elif engine == "hf":
        emissions = inference_hf(prompt, model_path, tokenizer, seed=seed, lora=lora)
    else:
        raise ValueError(f"No inference function defined for engine {engine}")
    return emissions

def main():
    model_config_path = "model_config.json"
    parser = ArgumentParser()
    parser.add_argument("document", type=str, help="File path or URL to the report from which emission values should be extracted")
    parser.add_argument("--model_path", default="models/Mistral-7B-Instruct-v0.2", help="Path to a directory containing the GGUF model and huggingface tokenizer. Mutually exclusive with --model.")
    parser.add_argument("--model_context_size", default=32768, help="The context size of the model. Can be retrieved by inspecting the specific model or config files. Must be used together with --model_path. Mutually exclusive with --model.")
    parser.add_argument("--model", default="", help=f"Given the model name, loads the model_path and model_context_size automatically from the {model_config_path} file. Mutually exclusive with --model_path and --model_context_size.")
    parser.add_argument("--prompt_template", default=None, help="The plaintext prompt template to use for prompt contruction. Falls back to prompt-templates/simple.txt.")
    parser.add_argument("--prompt_output_path", default="", help="If specified, saves the input prompt to this file")
    parser.add_argument("--grammars_dir", default="grammars", help="Grammar is stored in this dir. Set to /tmp if persistance not needed.")
    parser.add_argument("--extraction_mode", default="xhtml", choices=["xhtml", "text"], help="Whether to extract plain text or semi-semantic xhtml from document pages.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_group_neighbour_size", type=int, default=8)
    parser.add_argument("--max_group_window_size", type=int, default=2048)
    parser.add_argument("--lora", default="", help="Path to a LoRA to use together with the model.")
    parser.add_argument("--engine", default="llama.cpp", choices=["llama.cpp", "hf"], help="Inference engine to use.")
    args = parser.parse_args()

    if args.model:
        with open(model_config_path, "r") as json_file:
            model_config = load(json_file)[args.model]
    else:
        model_config = {
            "model_path": args.model_path,
            "context_size": args.model_context_size,
        }

    emissions = extract_emissions(document=args.document, model_path=model_config["model_path"], prompt_template=args.prompt_template, model_context_size=model_config["context_size"], prompt_output_path=args.prompt_output_path, grammars_dir=args.grammars_dir, extraction_mode=args.extraction_mode, seed=args.seed, max_group_neighbour_size=args.max_group_neighbour_size, max_group_window_size=args.max_group_window_size, lora=args.lora, engine=args.engine)
    print(emissions.model_dump_json())


if __name__ == "__main__":
    main()
