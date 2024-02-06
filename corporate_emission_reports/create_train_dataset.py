from argparse import ArgumentParser
from glob import iglob
from json import load, loads, dumps
from os import remove
from os.path import basename, normpath, isdir, isfile, join, splitext
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from corporate_emission_reports.extraction import extract_chunks_from_document
from corporate_emission_reports.pydantic_types import Emissions


def Int64_to_optional(v: pd.Int64Dtype) -> Optional[int]:
    if pd.isna(v):
        return None
    return int(v)
    

def row_to_emissions(row):
    emissions = Emissions(scope_1=Int64_to_optional(row["scope_1"]), scope_2=Int64_to_optional(row["scope_2"]), scope_3=Int64_to_optional(row["scope_3"]), sources=map(Int64_to_optional, row["sources"]))
    return emissions.model_dump_json()


def create_train_dataset(dataset_path="data/emissions_sft.jsonl", predictions_source="output_train/Mixtral-8x7B-Instruct-v0.1", generated_prompts_dir="prompts/generated/Mixtral-8x7B-Instruct-v0.1", force_prompt_regeneration=False, prompt_template="prompts/templates/simple.jsonl", documents_dir="pdfs_train", extraction_mode="xhtml"):
    stub_content = '{"scope_1":-1,"scope_2":-1,"scope_3":-1,"sources":[-1]}'
    if isfile(dataset_path):
        remove(dataset_path)
    
    if splitext(predictions_source)[1] == ".parquet":
        table = pq.read_table(predictions_source, columns=["id", "scope_1", "scope_2", "scope_3", "sources"])
        responses = {row["id"]: row_to_emissions(row) for _, row in table.to_pandas().iterrows()}
    elif isdir(predictions_source):
        responses = {}
        for output in sorted(iglob(join(predictions_source, "*.json"))):
            uid = splitext(basename(normpath(output)))[0]
            with open(output, "r") as f:
                response = f.read()
            responses[uid] = response
    else:
        raise ValueError("predictions_source must be parquet or directory of json files")
    
    for uid, response in responses.items():
        print(uid)
        if response == stub_content:
            continue  # ignore stubs
        gen_prompt_file = join(generated_prompts_dir, uid + ".txt")
        # Use generated prompt if available
        if not force_prompt_regeneration and isfile(gen_prompt_file):
            with open(gen_prompt_file, "r") as  f:
                prompt = f.read()
            # remove instruction template so the prompt can be used with other templates
            prompt = prompt[len("[INST] "):-len(" [INST]\n")]
            # ignore empty reports
            if prompt.endswith("</FORMAT>"):
                continue
        else:
            document = join(documents_dir, uid + ".pdf")
            if not isfile(document):
                continue
            with open(prompt_template, "r") as prompt_file:
                line = prompt_file.readline()
            prompt = loads(line)["content"]
            user_message = extract_chunks_from_document(document, mode=extraction_mode)
            if not user_message:
                continue
            prompt += "\n" + user_message[:-1]
        data = {"prompt": prompt, "completion": response}
        with open(dataset_path, "a") as f:
            f.write(dumps(data) + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", default="data/emissions_sft.jsonl", help="Dataset will be saved to this path.")
    parser.add_argument("--predictions_source", default="output_train/Mixtral-8x7B-Instruct-v0.1", help="Where to get the model predictions for the `completion` field from. Can be directory of json files containing the prediction for each report or a parquet file containing the `id`, `scope_1`, `scope_2`, `scope_3` and `sources` fields for all reports.")
    parser.add_argument("--generated_prompts_dir", default="prompts/generated/Mixtral-8x7B-Instruct-v0.1", help="The directory containing the cached prompts from the output generation. If set, uses available cached prompts instead of regenerating them. NOTE: only use this if the Mistral/Llama-2 instruct template was used for the cached prompts. Not used if --force_prompt_regeneration is set.")
    parser.add_argument("--force_prompt_regeneration", action="store_true", help="Whether to ignore cached generated prompt and regenerate the prompts. NOTE: this must be set if the Mistral/Llama-2 instruct template was not used for the cached prompts.")
    parser.add_argument("--prompt_template", default="prompts/templates/simple.jsonl", help="The prompt template in ChatML format to use for prompt generation. This must be set if --force_prompt_regeneration is set.")
    parser.add_argument("--documents_dir", default="pdfs_train", help="Directory containing the documents from which emission values were extracted. Used for promp generation. This must be set if --force_prompt_regeneration is set.")
    parser.add_argument("--extraction_mode", default="xhtml", choices=["xhtml", "text"], help="Whether to extract plain text or semi-semantic xhtml from document pages. This must be set if --force_prompt_regeneration is set.")
    args = parser.parse_args()
    create_train_dataset(dataset_path=args.dataset_path, predictions_source=args.predictions_source, generated_prompts_dir=args.generated_prompts_dir, force_prompt_regeneration=args.force_prompt_regeneration, prompt_template=args.prompt_template, documents_dir=args.documents_dir, extraction_mode=args.extraction_mode)


if __name__ == "__main__":
    main()
