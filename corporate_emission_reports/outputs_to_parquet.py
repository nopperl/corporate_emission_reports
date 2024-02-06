#!/usr/bin/env python
from argparse import ArgumentParser
from glob import iglob
from os.path import basename, normpath, join, splitext
from json import load

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def output_dir_to_parquet(dirname, output_file=None):
    data = []
    for filename in sorted(iglob(join(dirname, "*.json"))):
        with open(filename, "r") as json_file:
            obj = load(json_file)
            obj["id"] = splitext(basename(normpath(filename)))[0]
            # clip unusually large or small numbers
            obj = {k: min(max(v, -1), 1000000000) if type(v) in [int, float] else v for k, v in obj.items()}
            data.append(obj)
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    if not output_file:
        output_file = normpath(dirname) + ".parquet"
    pq.write_table(table, output_file)
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("dirname", type=str, help="A directory containing model output JSON files")
    parser.add_argument("--output_file", default="", help="Path to the output parquet file. By default, the output file is written next to dirname")
    args = parser.parse_args()
    output_dir_to_parquet(args.dirname, output_file=args.output_file)


if __name__ == "__main__":
    main()
