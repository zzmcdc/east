import mxnet as mx
import numpy as np
import time
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='train east with random shape')
  parser.add_argument('--symbol_file', type=str, default='east-symbol.json',
                      help='base network name which serves as feature extraction base.')
  parser.add_argument('--param_file', type=str, default='east-0000.params' ,help='param-file path')
  parser.add_argument('--dtype', type=str, default='float32')
  parser.add_argument('--gpu_id', type=int,default=0)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  sym = mx.sym.load(args.symbol_file)
  net = mx.mod.Module(symbol=sym, data_names=['data'], label_names=[], context=mx.gpu(args.gpu_id))
  net.bind(for_training=False, data_shapes=[('data', (1, 3, 512, 512))])
  net.load_params(args.param_file)
  im_fake = np.zeros(shape=(1, 3, 512, 512), dtype=args.dtype)
  for _ in range(10):
    net.forward(data_batch=mx.io.DataBatch(data=[mx.nd.array(im_fake)]), is_train=False)
    mx.nd.waitall()
  start = time.time()
  for _ in range(1000):
    net.forward(data_batch=mx.io.DataBatch(data=[mx.nd.array(im_fake)]), is_train=False)
    mx.nd.waitall()
  end = time.time()
  print('using average time:{:.3f}'.format((end - start) / 1000))
