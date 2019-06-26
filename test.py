import mxnet as mx
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='load mode and test speed')
    parser.add_argument('--sym', type=str, help='sym file path')
    parser.add_argument('--params', type=str, help='params file path')
    parser.add_argument('--width', type=int, help='width test')
    parser.add_argument('--height', type=int, help='height')
    args = parser.parse_args()
    return args

args = parse_args()

sym = mx.sym.load(args.sym)
net = mx.mod.Module(symbol=sym,data_names=['data'],label_names=[], )

