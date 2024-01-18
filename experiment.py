#!/usr/bin/env python
from argparse import ArgumentParser
from glob import iglob
from json import load, loads
from os import makedirs
from os.path import basename, normpath, isfile, join, splitext
from traceback import print_exc

from pydantic_types import Emissions

from inference import extract_emissions


def process_documents(documents_dir="pdfs", prompt_template="prompts/templates/simple.jsonl", model_path="models/Mistral-7B-Instruct-v0.2", model_context_size=32768, outputs_dir="outputs", generated_prompts_dir="prompts/generated", grammars_dir="grammars", extraction_mode="xhtml", seed=123, max_group_neighbour_size=64, max_group_window_size=1024):
    model_name = normpath(model_path).split('/')[-1]
    output_dir = join(outputs_dir, model_name)
    generated_prompt_dir = join(generated_prompts_dir, model_name)
    makedirs(output_dir, exist_ok=True)
    makedirs(generated_prompt_dir, exist_ok=True)
    makedirs(grammars_dir, exist_ok=True)
    
    for document in sorted(iglob(join(documents_dir, "*.pdf"))):
        uid = splitext(basename(normpath(document)))[0]
        output_path = join(output_dir, f"{uid}.json")
        print(f"Processing report id={uid} using model {model_name}. Output will be at {output_path}")
    
        if isfile(output_path):
            print(f"Report id={uid} already processed using model {model_name}. Skipping...")
            continue
        
        prompt_output_path = join(generated_prompt_dir, f"{uid}.txt")
        try:
            emissions = extract_emissions(document=document, model_path=model_path, prompt_template=prompt_template, model_context_size=model_context_size, prompt_output_path=prompt_output_path, grammars_dir=grammars_dir, extraction_mode=extraction_mode, seed=seed, max_group_neighbour_size=max_group_neighbour_size, max_group_window_size=max_group_window_size)
            output_json = emissions.model_dump_json()
        except:
            print_exc()
            print(f"Processing report={uid} failed. Skipping...")
            output_json = '{"scope_1":-1,"scope_2":-1,"scope_3":-1,"sources":[-1]}'

        with open(output_path, "w") as f:
            f.write(output_json)


def main():
    model_config_path = "model_config.json"
    parser = ArgumentParser()
    parser.add_argument("--documents_dir", default="pdfs", help="Directory containing the documents from which emission values should be extracted")
    parser.add_argument("--model_path", default="models/Mistral-7B-Instruct-v0.2", help="Path to a directory containing the GGUF model and huggingface tokenizer. Mutually exclusive with --model.")
    parser.add_argument("--model_context_size", default=32768, help="The context size of the model. Can be retrieved by inspecting the specific model or config files. Must be used together with --model_path. Mutually exclusive with --model.")
    parser.add_argument("--model", default="", help=f"Given the model name, loads the model_path and model_context_size automatically from the {model_config_path} file. Mutually exclusive with --model_path and --model_context_size.")
    parser.add_argument("--prompt_template", default="prompts/templates/simple.jsonl")
    parser.add_argument("--output_dir", default="outputs", help="saves the extraction output JSON files to this dir")
    parser.add_argument("--prompt_output_dir", default="prompts/generated", help="Saves the input prompts to this dir")
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

    process_documents(documents_dir=args.documents_dir, model_path=model_config["model_path"], prompt_template=args.prompt_template, model_context_size=model_config["context_size"], outputs_dir=args.output_dir, generated_prompts_dir=args.prompt_output_dir, grammars_dir=args.grammars_dir, extraction_mode=args.extraction_mode, seed=args.seed, max_group_neighbour_size=args.max_group_neighbour_size, max_group_window_size=args.max_group_window_size)


if __name__ == "__main__":
    main()
