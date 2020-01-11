import argparse
import random

import numpy as np
import json


def parseJson(args, jsonPath):
    with open(jsonPath, "r") as fp:
        jsonConfs = json.load(fp)

    for key, value in jsonConfs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            assert False, "the key is not in added arguments"
    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # Training Hyper-parameters
    parser.add_argument("--numWorkers", type=int, default=0)
    parser.add_argument('--batchSize', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument("--schedulerGamma", type=float, default=0.5)
    parser.add_argument("--schedulerStepSize", type=int, default=1000)

    # Debugging
    parser.add_argument('--expFolder', type=str, default='../exps/first')
    parser.add_argument("--json-conf", type=str, default=None)

    # Training configuration

    parser.add_argument("--trainFolder", default="../data/coco/coco_train2014", type=str)
    parser.add_argument("--valFolder", default="../data/coco/coco_val2014", type=str)
    parser.add_argument("--valStep", type=int, default=500, help='Number of steps to run validation.')
    parser.add_argument("--logStep", type=int, default=10, help='Number of steps to log.')

    # Parse the arguments.
    args = parser.parse_args()
    if args.json_conf:
        args = parseJson(args, args.json_conf)

    print(args)
    return args


args = parse_args()
