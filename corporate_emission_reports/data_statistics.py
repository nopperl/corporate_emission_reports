#!/usr/bin/env python
from argparse import ArgumentParser
from glob import iglob
from os.path import join
from statistics import median, mean

import pandas as pd
from transformers import AutoTokenizer


def get_text_token_length(text):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    text_tokenized = tokenizer.encode(text)
    return len(text_tokenized)
    

def process_directory(dirname):
    lengths = []
    for filename in iglob(join(dirname, "*.txt")):
       with open(filename, 'r') as file:
           text = file.read()
       lengths.append(get_text_token_length(text))
    return {"token_length": {"max": max(lengths), "min": min(lengths), "mean": mean(lengths), "median": median(lengths)}}


def main():
    parser = ArgumentParser()
    parser.add_argument("--generated_prompts_dir", default="prompts/generated/Mistral-7B-Instruct-v0.2", help="Directory to process")
    parser.add_argument("--output_file", default="report/tables/token.tex")
    args = parser.parse_args()
    data = process_directory(args.generated_prompts_dir)
    print(data)
    df = pd.DataFrame(data).T
    if args.output_file:
        with open(args.output_file, "w") as tex_file:
            df.to_latex(buf=tex_file, escape=True, float_format="%.2f")


if __name__ == '__main__':
    main()

