#!/usr/bin/env python
from argparse import ArgumentParser
from glob import iglob
from os import makedirs
from os.path import basename, normpath, join, splitext

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def evaluate_key_for_model(pred_file, gt_file, key="scope_1", mode="strict"):
    if key == "scope_2":
        key_gt = pq.read_table(gt_file, columns=["scope_2_market"])[0].to_numpy()
        # TODO: maybe merge market and location instead, pred == gt_market | pred == gt_location
        market_emissions_na = np.isnan(key_gt)
        key_gt[market_emissions_na] = pq.read_table(gt_file, columns=["scope_2_location"])[0].to_numpy()[market_emissions_na]
    else:
        key_gt = pq.read_table(gt_file, columns=[key])[0].to_numpy()
    key_pred = pq.read_table(pred_file, columns=[key])[0].to_numpy()
    # TODO: check if ids are sorted/in same order
    # TODO: consider if failed reports should negatively affect accuracy after all
    valid_mask = key_pred != -1  # ignore stub outputs of failed reports
    key_pred = key_pred[valid_mask]
    key_gt = key_gt[valid_mask]
    both_na_mask = np.isnan(key_pred) & np.isnan(key_gt)
    correct = both_na_mask.sum()
    if mode == "strict":
        correct += np.isclose(key_pred.round(), key_gt.round()).sum()  # isclose due to float values
    elif mode == "tolerant":
        correct += np.isclose(key_pred, key_gt, rtol=.1, atol=1).sum()
    elif mode == "graceful":
        base_unit = np.vectorize(lambda x: x.rstrip("0"))
        correct = np.equal(base_unit(key_pred.round().astype(int).astype(str)), base_unit(key_gt.round().astype(int).astype(str))).sum()
    else:
        raise ValueError("mode must be strict, tolerant or graceful")
    
    accuracy = correct / len(key_pred)
    return accuracy


def evaluate_sources_for_model(pred_file, gt_file, pred_key, gt_keys):
    if "scope_2_page" in gt_keys:
        scope_2_idx = gt_keys.index("scope_2_page")
        gt_keys[scope_2_idx] = "scope_2_location_page"
        gt_keys.insert(scope_2_idx, "scope_2_market_page")
    key_pred = pq.read_table(pred_file, columns=[pred_key])[0].to_pylist()
    key_gt = pq.read_table(gt_file, columns=gt_keys)[0].to_pylist()
    # TODO: check if ids are sorted/in same order
    correct = 0
    skipped = 0
    for i in range(len(key_pred)):
        if len(key_pred) == 1 and key_pred[0] == -1:
            skipped += 1 # TODO: consider if failed reports should negatively affect accuracy after all
            continue
        intersection = set(key_pred[i]).intersection(key_gt[i])
        if len(intersection) > 0:
            correct += 1
    accuracy = (correct + skipped) / len(key_pred)
    return accuracy

def evaluate(pred_dir="outputs", gt_file="data/corp_emissions.parquet", evaluation_type="values", mode="strict", keys=None, output_dir="report/tables"):
    if keys is None:
        keys = ["scope_1", "scope_2", "scope_3"]
    data = {}
    makedirs(output_dir, exist_ok=True)
    
    if evaluation_type == "sources":
        keys = [key + "_page" for key in keys if not key.endswith("_page")]
    for pred_file in iglob(join(pred_dir, "*.parquet")):
        model_name = splitext(basename(normpath(pred_file)))[0]
        if evaluation_type == "values":
            data[model_name] = {}
            for key in keys:
                data[model_name][key] = evaluate_key_for_model(pred_file, gt_file, key, mode=mode)
        elif evaluation_type == "sources":
            data[model_name] = {"accuracy": evaluate_sources_for_model(pred_file, gt_file, pred_key=evaluation_type, gt_keys=keys)}
    print(data)

    if output_dir: 
        latex_filename = join(output_dir, evaluation_type)
        latex_filename += f"-{mode}.tex" if evaluation_type == "values" else ".tex"
        
        df = pd.DataFrame(data).T
        if df.shape[1] > 1:
            df["average"] = df.mean(axis=1)
        #df.columns = [c.replace("_", "\_") for c in df.columns]
        with open(latex_filename, "w") as tex_file:
            df.to_latex(buf=tex_file, escape=True, float_format="%.2f")


def main():
    parser = ArgumentParser()
    parser.add_argument("--prediction_dir", default="outputs", help="Directory containg parquet files with model outputs")
    parser.add_argument("--ground_truth_file", default="data/corp_emissions.parquet", help="Parquet file containing the ground truth values")
    parser.add_argument("--evaluation_type", type=str, choices=["values", "sources"], default="values")
    parser.add_argument("--mode", type=str, choices=["strict", "tolerant", "graceful"], default="strict", help="Ignored for --evaluation_type sources")
    parser.add_argument("--output_dir", default="report/tables", help="Directory to store LaTeX tables for the metrics")
    parser.add_argument("--keys", default=["scope_1", "scope_2", "scope_3"], nargs="+")
    args = parser.parse_args()
    evaluate(pred_dir=args.prediction_dir, gt_file=args.ground_truth_file, evaluation_type=args.evaluation_type, mode=args.mode, keys=args.keys, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

