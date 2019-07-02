import mxnet as mx
from mxnet import gluon
import gluoncv as gcv
from mxnet.gluon import nn
import numpy as np
from feature import FPNFeatureExpander
from mxnet import autograd
from mxboard import SummaryWriter


class East(nn.HybridBlock):

  def __init__(self, base_model, outputs, text_scale=512, ctx=mx.cpu(), pretrained_base=True, **kwargs):
    super(East, self).__init__()
    self.text_scale = text_scale
    weight_init = mx.init.Xavier(factor_type="in", magnitude=2.34)
    with self.name_scope():
      self.features = FPNFeatureExpander(network=base_model, outputs=outputs,pretrained=pretrained_base, ctx=ctx, **kwargs)

      self.score_branch = nn.Conv2D(1, 1, activation='sigmoid')
      self.geo_branch = nn.Conv2D(4, 1, activation='sigmoid')
      self.theta_branch = nn.Conv2D(1, 1, activation='sigmoid')

  def hybrid_forward(self, F, x, **kwargs):
    #x = F.Cast(x, dtype='float16')
    x = self.features(x)
    #x = F.Cast(x, dtype='float32')
    score_map = self.score_branch(x)
    geo_map = self.geo_branch(x) * self.text_scale

    angle_map = (self.theta_branch(x) - 0.5) * np.pi / 2.
    geometry_map = F.Concat(geo_map, angle_map, dim=1)

    return score_map, geometry_map


def get_east_resnet50(**kwargs):
  net = East(base_model='resnet50_v1d',
            outputs=['pool0_fwd','layers2_bottleneckv1b0__plus0','layers3_bottleneckv1b1__plus0' ,'layers4_relu8_fwd'], **kwargs) #layers4_bottleneckv1b2__plus0
            #outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd', 'layers4_relu8_fwd'], **kwargs)

  return net


def get_east_mobilenet(**kwargs):
  net = East(base_model='mobilenetv2_1.0',
             outputs=['features_linearbottleneck3_relu60_relu6', 'features_linearbottleneck6_relu60_relu6',
                      'features_linearbottleneck9_relu60_relu6', 'features_linearbottleneck16_relu60_relu6'], **kwargs)
  return net


_models = {
  'resnet50': get_east_resnet50,
  'mobilenet': get_east_mobilenet,
}


def get_model(name, **kwargs):
  name = name.lower()
  if name not in _models:
    err_str = '"%s" is not among the following model list:\n\t' % (name)
    err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
    raise ValueError(err_str)
  net = _models[name](**kwargs)
  return net


if __name__ == '__main__':
  net = get_model('resnet50', pretrained_base=True)
  net.hybridize()
  net.initialize()
  with autograd.train_mode():
    x = mx.nd.array([np.random.normal(size=(3, 512, 512))])
    net(mx.nd.random.uniform(low=0, high=1, shape=(1, 3, 512, 512)))
    with SummaryWriter(logdir='./logs') as sw:
      sw.add_graph(net)
