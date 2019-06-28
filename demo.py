import mxnet as mx
from network import get_model
import cv2
import numpy as np
from utils import restore_rectangle
import lanms
import argparse
import ipdb


def parse_args():
    parser = argparse.ArgumentParser(
        description='train east with random shape')
    parser.add_argument('--network', type=str, default='resnet50',
                        help='base network name which serves as feature extraction base.')
    parser.add_argument('--param_file', type=str, help='param-file pathj')
    parser.add_argument('--data_shape', type=int, default=512,
                        help='input data shape for evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='using gpu id')
    parser.add_argument('--image', type=str, help='image path for train')
    parser.add_argument('--score_map_threshold', type=float,
                        default=0.9, help='score threshold')
    parser.add_argument('--box_threshold', type=float,
                        default=0.1, help='box threshold')
    parser.add_argument('--nms_threshold', type=float,
                        default=0.1, help='nms threshold')
    args = parser.parse_args()
    return args


def load_image(img, data_shape):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    im_padded = cv2.resize(img, (data_shape[0], data_shape[1]))
    img = mx.nd.array(im_padded)
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)

    return img


if __name__ == '__main__':
    args = parse_args()
    net = get_model(name=args.network, pretrained_base=False,
                    text_scale=args.data_shape, use_upsample=False, use_deconv=True)
    net.initialize()
    for item in net.features.collect_params().items():
        if item[0].split('_')[-1] not in ['gamma', 'beta', 'mean', 'var']:
            item[1].cast('float16')
    net.load_parameters(args.param_file)
    net.collect_params().reset_ctx(mx.gpu(0))
    net.hybridize()
    img = cv2.imread(args.image)
    img_input = load_image(img, (args.data_shape, )*2)
    h, w, c = img.shape

    w_ration = args.data_shape/w
    h_ration = args.data_shape/h
    img_input = mx.nd.expand_dims(img_input, 0)
    img_input = img_input.as_in_context(mx.gpu(0))
    #img_input = mx.nd.cast(img_input)

    score_map_thresh = args.score_map_threshold
    box_thresh = args.box_threshold
    nms_thres = args.nms_threshold

    score_map, geo_map = net(img_input)
    score_map = score_map.asnumpy()
    geo_map = geo_map.asnumpy()
    score_map = np.copy(score_map[0, 0, :, :])
    geo_map = np.transpose(np.copy(geo_map[0, :, :, ]), (1, 2, 0))
    xy_text = np.argwhere(score_map > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    text_box_restored = restore_rectangle(
        xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    ipdb.set_trace()
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape(
            (-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= w_ration
        boxes[:, :, 1] /= h_ration
    for box in boxes:
        for i in range(4):
            cv2.line(img, (box[i][0], box[i][1]), (box[(i + 1) % 4][0],
                                                   box[(i + 1) % 4][1]), color=(255, 0, 0), thickness=2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    net.export('east',0)
