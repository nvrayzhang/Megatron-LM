# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,
                                             os.path.pardir)))
import time
import torch
from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset

from utils import zng, has_chinese

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            Encoder.splitter = zng
        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, data):
        if self.args.json_by_line:
          data = json.loads(data)

        num_tokens = 0
        ids = {}
        for key in self.args.json_keys:
            text = data[key]

            # JQ: Skip non-chinese text
            if not has_chinese(text):
              continue

           #print(f"Text: {text}")
            doc_ids = []   # a list of list
            # 1. Split text into sentences
            # 2. Tokenize sentence into IDs
            for sentence in Encoder.splitter(text):
                # TODO remove any sentence with symbols only
                sentence_ids = Encoder.tokenizer.tokenize(sentence)

                if self.args.debug:
                  print(f"Sentence: {sentence}")
                  decoded = Encoder.tokenizer.decode(sentence_ids); print(f"Decode: {decoded}")

                if len(sentence_ids) > 0:
                    num_tokens += len(sentence_ids)
                    doc_ids.append(sentence_ids)

            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids

        return ids, num_tokens


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    # JQ: select file format
    group.add_argument('--json-by-line', action='store_true',
                       help='Each line is a json object')
    # JQ: debug output
    group.add_argument('--debug', action='store_true',
                       help='Each line is a json object')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    """ JQ: not needed by chinese
    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)
    """

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')
    if not args.json_by_line:
      # File is a list of objs
      fin = json.load(fin)
      if args.debug:
        fin = fin[:args.workers]
    encoded_docs = pool.imap(encoder.encode, fin, 128)
    #encoded_docs = map(encoder.encode, fin)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        # Create a class MMapIndexedDatasetBuilder(object):
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                               impl=args.dataset_impl,
                                               vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_tokens_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, tokens_processed) in enumerate(encoded_docs, start=1):
        # doc is a dict: ["text"] = a list of list
        total_tokens_processed += tokens_processed
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                # Convert tensor to numpy bytes, save to file
                builders[key].add_item(torch.IntTensor(sentence))
            # Add a index mark for doc
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_tokens_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed:.2f} docs/s, tokens: {mbs:.2f} Million/s).",
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])
        print("Save output to {} and {}".format(
            output_bin_files[key], output_idx_files[key])

    print("Total number of tokens: {}".format(total_tokens_processed))

if __name__ == '__main__':
    main()
