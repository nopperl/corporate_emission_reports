#!/usr/bin/env python
from argparse import ArgumentParser
from json import load, loads
from math import ceil
import os
from os.path import isfile, join
from tempfile import NamedTemporaryFile
from typing import Optional
from subprocess import check_output
from urllib.request import urlretrieve
from urllib.parse import urlparse

from pydantic_types import Emissions, pydantic_model_to_grammar
from transformers import AutoTokenizer
from extraction import extract_chunks_from_document

from download_documents import download_document


def try_download_document(document) -> Optional[str]:
     if urlparse(document).scheme not in ["https", "http", "ftp"]:
         return None
     return urlretrieve(document)[0]


def inference_llamacpp(prompt_token_len, prompt_file, model_path, eos, model_context_size=32768, grammar_file="", seed=123, max_group_neighbour_size=16, max_group_window_size=1024):
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
    result = check_output(["sh", "inference-llamacpp.sh"], env=env_vars)
    output = result.decode()
    # Strip eos token if printed
    if output[-len(eos):] == eos:
        output = output[:-len(eos)]
    print("Output:", output)
    return Emissions.model_validate_json(output, strict=True)


def extract_emissions(document, model_path, prompt_template, model_context_size=32768, prompt_output_path=None, grammars_dir="grammars", extraction_mode="xhtml", seed=123, max_group_neighbour_size=8, max_group_window_size=2048) -> Emissions:
    if not isfile(document):
        document = try_download_document(document)
        if document is None:
            raise ValueError("document needs to be a path to an existing file or an URL to a network object")

    grammar = pydantic_model_to_grammar(Emissions)
    grammar_path = f"{grammars_dir}/emissions_json.gbnf"
    with open(grammar_path, "w") as f:
        f.write(grammar)

    prompt = []
    with open(prompt_template, "r") as prompt_file:
        for line in prompt_file:
            prompt.append(loads(line))

    user_message = extract_chunks_from_document(document, mode=extraction_mode)
    prompt.append({"role": "user", "content": user_message})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    bos_len = 0 if tokenizer.bos_token is None else len(tokenizer.bos_token)
    prompt_text = prompt_text[bos_len:]  # llama.cpp automatically adds bos
    prompt_text += ' I have extracted the Scope 1, 2 and 3 emission values from the document, converted them into metric tons and put them into the following json object:\n```json\n'
    if prompt_output_path is None or prompt_output_path == "":
        prompt_file = NamedTemporaryFile("w", delete=False)
        prompt_output_path = prompt_file.name
    else:
        prompt_file = open(prompt_output_path, "w")
    with prompt_file:
        prompt_file.write(prompt_text)

    prompt_tokenized = tokenizer.apply_chat_template(prompt, return_tensors="np")
    token_length = prompt_tokenized.shape[1]
    eos_token = tokenizer.eos_token
    if "QWenTokenizer" in str(type(tokenizer)):
        token_length = int(token_length * 1.2)  # HF and llama.cpp use tokenizers that produce different lengths for QWen
        eos_token = "[PAD151643]"  # llama.cpp tokenizer uses this as eos
    emissions = inference_llamacpp(token_length, prompt_output_path, model_path, eos_token, model_context_size=model_context_size, grammar_file=grammar_path, seed=seed, max_group_neighbour_size=max_group_neighbour_size, max_group_window_size=max_group_window_size)
    return emissions

def main():
    model_config_path = "model_config.json"
    parser = ArgumentParser()
    parser.add_argument("document", type=str, help="File path or URL to the report from which emission values should be extracted")
    parser.add_argument("--model_path", default="models/Mistral-7B-Instruct-v0.2", help="Path to a directory containing the GGUF model and huggingface tokenizer. Mutually exclusive with --model.")
    parser.add_argument("--model_context_size", default=32768, help="The context size of the model. Can be retrieved by inspecting the specific model or config files. Must be used together with --model_path. Mutually exclusive with --model.")
    parser.add_argument("--model", default="", help=f"Given the model name, loads the model_path and model_context_size automatically from the {model_config_path} file. Mutually exclusive with --model_path and --model_context_size.")
    parser.add_argument("--prompt_template", default="prompts/templates/simple.jsonl")
    parser.add_argument("--prompt_output_path", default="", help="If specified, saves the input prompt to this file")
    parser.add_argument("--grammars_dir", default="grammars", help="Grammar is stored in this dir. Set to /tmp if persistance not needed.")
    parser.add_argument("--extraction_mode", default="xhtml", choices=["xhtml", "text"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_group_neighbour_size", type=int, default=8)
    parser.add_argument("--max_group_window_size", type=int, default=2048)
    args = parser.parse_args()

    if args.model:
        with open(model_config_path, "r") as json_file:
            model_config = load(json_file)[args.model]
    else:
        model_config = {
            "model_path": args.model_path,
            "context_size": args.model_context_size,
        }

    emissions = extract_emissions(document=args.document, model_path=model_config["model_path"], prompt_template=args.prompt_template, model_context_size=model_config["context_size"], prompt_output_path=args.prompt_output_path, grammars_dir=args.grammars_dir, extraction_mode=args.extraction_mode, seed=args.seed, max_group_neighbour_size=args.max_group_neighbour_size, max_group_window_size=args.max_group_window_size)
    print(emissions.model_dump_json())


if __name__ == "__main__":
    main()
