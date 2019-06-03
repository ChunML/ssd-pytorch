import torch


def compute_area(top_left, bot_right):
    # top_left: N x 2
    # bot_right: N x 2
    hw = torch.clamp(bot_right - top_left, min=0.0)
    area = hw[..., 0] * hw[..., 1]

    return area


def compute_iou(boxes_a, boxes_b):
    # boxes_a: N1 x 4
    # boxes_b: N2 x 4

    # return overlap: N1 x N2
    # boxes_a => N1 x 1 x 4
    # boxes_b => 1 x N2 x 4
    boxes_a = boxes_a.unsqueeze(1)
    boxes_b = boxes_b.unsqueeze(0)
    top_left = torch.max(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = torch.min(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap


def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
    # default_boxes: N1 x 4
    # boxes: N2 x 4
    transformed_default_boxes = transform_center_to_corner(default_boxes)
    iou = compute_iou(transformed_default_boxes, gt_boxes)
    best_gt_iou, best_gt_idx = iou.max(1)
    best_default_iou, best_default_idx = iou.max(0)

    for gt_idx, default_idx in enumerate(best_default_idx):
        best_gt_idx[default_idx] = gt_idx

    best_gt_iou.index_fill_(0, best_default_idx, 2)

    gt_confs = gt_labels[best_gt_idx]
    gt_confs[best_gt_iou < iou_threshold] = 0
    gt_boxes = gt_boxes[best_gt_idx]

    gt_locs = encode(default_boxes, gt_boxes)

    return gt_confs, gt_locs


def encode(default_boxes, boxes, variance=[0.1, 0.2]):
    # default_boxes: N1 x 4
    # boxes: N2 x 4

    # Convert boxes to (cx, cy, w, h) form
    transformed_boxes = transform_corner_to_center(boxes)

    locs = torch.cat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / (default_boxes[:, 2:] * variance[0]),
        torch.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]], dim=-1)

    return locs


def decode(default_boxes, locs, variance=[0.1, 0.2]):
    # default_boxes: N1 x 4
    # locs: B x N2 x 4
    locs = torch.cat([
        locs[..., :2] * variance[0] * default_boxes[:, 2:] + default_boxes[:, :2],
        torch.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], dim=-1)

    boxes = transform_center_to_corner(locs)

    return boxes


def transform_corner_to_center(boxes):
    # box: (B x ) N x 4
    center_box = torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], dim=-1)

    return center_box


def transform_center_to_corner(boxes):
    # boxes: (B x ) N x 4
    corner_box = torch.cat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], dim=-1)

    return corner_box


def compute_nms(boxes, scores, nms_threshold, limit=200):
    if boxes.size(0) == 0:
        return []
    selected = [0]
    _, idx = scores.sort(descending=True)
    idx = idx[:limit]
    boxes = boxes[idx]

    iou = compute_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        iou[:, ~next_indices] = 1.0

        if next_indices.sum().item() == 0:
            break

        selected.append(next_indices.argsort(descending=True)[0].item())

    return idx[selected]
