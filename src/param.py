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

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='../exps/first')
    parser.add_argument("--valTopK", type=int, default=None,
                        help="number of validation images which will be validated")

    parser.add_argument("--json-conf", type=str, default=None)

    # Training configuration
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    parser.add_argument("--train_img_dir", default="../data/coco/coco_train2014", type=str)
    parser.add_argument("--val_img_dir", default="../data/coco/coco_val2014", type=str)

    parser.add_argument("--val_step", type=int, default=500, help='Number of steps to run validation.')
    parser.add_argument("--log_step", type=int, default=10, help='Number of steps to log.')

    parser.add_argument("--finetune_detector", action="store_const", default=False, const=True)

    # Parse the arguments.
    args = parser.parse_args()
    if args.json_conf:
        args = parseJson(args, args.json_conf)

    print(args)
    # Bind optimizer class.
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
