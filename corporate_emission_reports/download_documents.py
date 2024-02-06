#!/usr/bin/env python
from argparse import ArgumentParser
from hashlib import sha256
from os import makedirs
from os.path import isfile, join
from traceback import print_exc
from urllib.request import Request, urlopen

from fake_useragent import UserAgent
import pyarrow.parquet as pq


def download_document(url, save_path, original_hash=None):
    hash_algorithm = sha256()
    if original_hash is not None and isfile(save_path):
        with open(save_path, "rb") as f:
            hash_algorithm.update(f.read())
            file_hash = hash_algorithm.hexdigest()
            if file_hash == original_hash:
                return True
            print(f"File at {save_path} (url: {url}) has conflicting hash. Downloading again...")
    headers = {"User-Agent": UserAgent().edge}
    with urlopen(Request(url, headers=headers), timeout=10) as f:
        data = f.read()
        hash_algorithm.update(data)
        file_hash = hash_algorithm.hexdigest()
        if original_hash is not None and file_hash != original_hash:
            print(f"File downloaded from {url} has conflicting hash. Not saving. Try running this function again.")
            return False
        with open(save_path, 'wb') as out_f:
            out_f.write(data)
    return True


def download_documents_from_dataset(dataset_path, document_dir="pdfs"):
    table = pq.read_table(dataset_path, columns=["id", "url", "sha256"])
    ids = table["id"].to_pylist()
    urls = table["url"].to_pylist()
    hashes = table["sha256"].to_pylist()
    makedirs(document_dir, exist_ok=True)
    failed_uids = []
    for i, uid in enumerate(ids):
        print(f"Downloading report id={uid} from {urls[i]}")
        save_path = join(document_dir, uid + ".pdf")
        try:
            success = download_document(urls[i], save_path, original_hash=hashes[i]) 
        except:
            print_exc()
            success = False
        if not success:
            failed_uids.append(uid)
            print(f"Failed to download report id={uid}. Try downloading the file using a browser.")
    if len(failed_uids) > 0:
        print(f"Failed to download reports for following ids: {failed_uids}. Try downloading the files using a browser.")


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--document_dir", default="pdfs")
    args = parser.parse_args()
    download_documents_from_dataset(args.dataset_path, args.document_dir)


if __name__ == "__main__":
    main()
