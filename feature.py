# pylint: disable=abstract-method
"""Feature extraction blocks.
Feature or Multi-Feature extraction is a key component in object detection.
Class predictor/Box predictor are usually applied on feature layer(s).
A good feature extraction mechanism is critical to performance.
"""
from __future__ import absolute_import

import mxnet as mx
from mxnet.base import string_types
from mxnet.gluon import HybridBlock, SymbolBlock
from mxnet.symbol import Symbol
from mxnet.symbol.contrib import SyncBatchNorm
from gluoncv.model_zoo import get_model

def _parse_network(network, outputs, inputs, pretrained, ctx, **kwargs):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or iterable of str
        The name of layers to be extracted as features.
    inputs : iterable of str
        The name of input datas.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).

    Returns
    -------
    inputs : list of Symbol
        Network input Symbols, usually ['data']
    outputs : list of Symbol
        Network output Symbols, usually as features
    params : ParameterDict
        Network parameters.
    """
    inputs = list(inputs) if isinstance(inputs, tuple) else inputs
    for i, inp in enumerate(inputs):
        if isinstance(inp, string_types):
            inputs[i] = mx.sym.var(inp)
        assert isinstance(inputs[i], Symbol), "Network expects inputs are Symbols."
    if len(inputs) == 1:
        inputs = inputs[0]
    else:
        inputs = mx.sym.Group(inputs)
    params = None
    prefix = ''
    if isinstance(network, string_types):
        network = get_model(network, pretrained=pretrained, ctx=ctx, **kwargs)
    if isinstance(network, HybridBlock):
        params = network.collect_params()
        prefix = network._prefix
        network = network(inputs)
    assert isinstance(network, Symbol), \
        "FeatureExtractor requires the network argument to be either " \
        "str, HybridBlock or Symbol, but got %s" % type(network)

    if isinstance(outputs, string_types):
        outputs = [outputs]
    assert len(outputs) > 0, "At least one outputs must be specified."
    outputs = [out if out.endswith('_output') else out + '_output' for out in outputs]
    outputs = [network.get_internals()[prefix + out] for out in outputs]
    return inputs, outputs, params




class FPNFeatureExpander(SymbolBlock):
    """Feature extractor with additional layers to append.
    This is specified for ``Feature Pyramid Network for Object Detection``
    which implement ``Top-down pathway and lateral connections``.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluon.model_zoo.vision if network is string.
        Convert to Symbol if network is HybridBlock.
    outputs : str or list of str
        The name of layers to be extracted as features
    num_filters : list of int e.g. [256, 256, 256, 256]
        Number of filters to be appended.
    use_1x1 : bool
        Whether to use 1x1 convolution
    use_upsample : bool
        Whether to use upsample
    use_elewadd : float
        Whether to use element-wise add operation
    use_p6 : bool
        Whther use P6 stage, this is used for RPN experiments in ori paper
    no_bias : bool
        Whether use bias for Convolution operation.
    norm_layer : HybridBlock or SymbolBlock
        Type of normalization layer.
    norm_kwargs : dict
        Arguments for normalization layer.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo if `True`.
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    inputs : list of str
        Name of input variables to the network.

    """

    def __init__(self, network, outputs,pretrained=True, ctx=mx.cpu(),
                 inputs=('data',)):
        inputs, f, params = _parse_network(network, outputs, inputs, pretrained, ctx)
        self.bn_eps = 1e-5  
        self.bn_mom = 0.997
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2.)

        # e.g. For ResNet50, the feature is :
        # outputs = ['stage1_activation2', 'stage2_activation3',
        #            'stage3_activation5', 'stage4_activation2']
        # with regard to [conv2, conv3, conv4, conv5] -> [C2, C3, C4, C5]
        # append more layers with reversed order : [P5, P4, P3, P2]
        f.reverse()
        umsample_size = [2048, 128, 64]
        num_outputs = [None, 128, 64, 32]
        g = [None, None, None, None]
        h = [None, None, None, None] 
        for i in range(4):
            if i == 0:
                h[i] = f[i]
            else:
                cur_data = mx.sym.Concat(*[g[i - 1], f[i]], dim=1)
                c1_1 = mx.sym.Convolution(data=cur_data, num_filter=num_outputs[i], kernel=(1, 1), no_bias=True, attr={'__init__': weight_init})
                c1_1 = mx.sym.BatchNorm(data=c1_1, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,momentum=self.bn_mom)
                
                c1_1 = mx.sym.Convolution(data=c1_1, num_filter=num_outputs[i], kernel=(3, 3), pad=(1, 1), no_bias=True,attr={'__init__': weight_init}) 
                h[i] = mx.sym.BatchNorm(data=c1_1, fix_gamma=False, use_global_stats=False, eps=self.bn_eps,momentum=self.bn_mom)
            if i <= 2:
                g[i] = mx.sym.Deconvolution(data=h[i],kernel=(3,3),stride=(2,2),pad=(1,1),adj=(1,1),num_filter=umsample_size[i]) 
   
            else:
                g[i] = mx.sym.Convolution(data=h[i], num_filter=num_outputs[i], kernel=(3, 3), pad=(1, 1), no_bias=True, attr={'__init__': weight_init})
                g[i] = mx.sym.BatchNorm(data=g[i], fix_gamma=False, use_global_stats=False, eps=self.bn_eps, momentum=self.bn_mom)
            
        super(FPNFeatureExpander, self).__init__(g[3], inputs, params)
