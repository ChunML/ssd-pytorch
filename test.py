import argparse
import torch
import torch.nn.functional as F
import numpy as np
from image_utils import ImageVisualizer
from data import create_dataloader
from network import create_ssd
from anchor import generate_default_boxes
from box_utils import decode, compute_nms
import cv2

NUM_CLASSES = 21
MEAN = [123, 117, 104]

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/VOCdevkit')
parser.add_argument('--save_image_dir', default='./images')
parser.add_argument('--pretrained_path', default='./models/ssd_epoch_23.pth')
parser.add_argument('--num_examples', default=40, type=int)
parser.add_argument('--max_num_boxes_per_class', default=200, type=int)
parser.add_argument('--score_thresh', default=0.6, type=float)
parser.add_argument('--nms_thresh', default=0.45, type=int)
parser.add_argument('--batch_size', default=1, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruct_image(img):
    """ Reconstruct processed image to export:
        1. Transpose from C, H, W to H, W, C
        2. Add back mean values
        3. Convert from RGB to BGR color

    Args:
        img: numpy array of shape (3, H, W)

    Returns:
        img: numpy array of shape (H, W, 3)
    """
    img = np.ascontiguousarray(img.transpose((1, 2, 0)))
    img += [123, 117, 104]
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def test(ssd, data, default_boxes):
    """ Execute test on one image
        then perform NMS algorithm
        then rescale boxes back to normal size

    Args:
        ssd: trained SSD model
        img: Torch array of shape (1, 3, H, W)
        default_boxes: Torch array of shape (num_default, 4)

    Returns:
        img: numpy array of (3, H, W)
        all_boxes: final boxes, numpy array of (num_boxes, 4)
        all_scores: final scores, numpy array of (num_boxes,)
        all_names: final class names, numpy array of (num_boxes,)
    """
    img, _, _ = data
    img = img.to(device)
    default_boxes = default_boxes.to(device)
    with torch.no_grad():
        out_confs, out_locs = ssd(img)
    out_confs = out_confs.squeeze(0)
    out_locs = out_locs.squeeze(0)
    out_boxes = decode(default_boxes, out_locs)
    out_labels = F.softmax(out_confs, dim=1)

    all_boxes = []
    all_scores = []
    all_names = []

    for c in range(1, NUM_CLASSES):
        cls_scores = out_labels[:, c]
        score_idx = cls_scores > args.score_thresh
        cls_boxes = out_boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        box_idx = compute_nms(
            cls_boxes, cls_scores,
            args.nms_thresh,
            args.max_num_boxes_per_class)

        cls_boxes = cls_boxes[box_idx]
        cls_scores = cls_scores[box_idx]
        cls_names = [c] * cls_boxes.size(0)

        all_boxes.append(cls_boxes)
        all_scores.append(cls_scores)
        all_names.extend(cls_names)

    all_boxes = torch.cat(all_boxes, dim=0)
    all_boxes *= 300
    all_scores = torch.cat(all_scores, dim=0)

    img = img.squeeze(0).cpu().numpy()
    all_boxes = all_boxes.cpu().numpy()
    all_scores = all_scores.cpu().numpy()
    all_names = np.array(all_names)

    return img, all_boxes, all_scores, all_names


if __name__ == '__main__':
    config = {
        'scales': [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
        'fm_sizes': [38, 19, 10, 5, 3, 1],
        'ratios': [(2,), (2, 3), (2, 3), (2, 3), (2,), (2,)]
    }

    default_boxes = generate_default_boxes(config)

    dataloader, info = create_dataloader(
        args.data_dir, args.batch_size,
        default_boxes, args.num_examples)

    ssd = create_ssd(NUM_CLASSES, 'ssd', args.pretrained_path)
    ssd.to(device)
    ssd.eval()

    visualizer = ImageVisualizer(
        info['idx_to_name'],
        save_dir=args.save_image_dir)

    for i, data in enumerate(dataloader):
        img, boxes, scores, names = test(ssd, data, default_boxes)
        img = reconstruct_image(img)
        img_name = 'image_{}.jpg'.format(i)
        visualizer.save_image(img, boxes, names, img_name)