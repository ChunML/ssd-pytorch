import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from image_utils import ImageVisualizer


class VOCDataset(Dataset):
    def __init__(self, root_dir, year, num_examples=-1):
        super(VOCDataset, self).__init__()
        self.idx_to_name = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])

        self.data_dir = os.path.join(root_dir, 'VOC{}'.format(year))
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.anno_dir = os.path.join(self.data_dir, 'Annotations')
        self.ids = list(map(lambda x: x[:-4], os.listdir(self.image_dir)))

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        filename = self.ids[index]
        img_path = os.path.join(self.image_dir, filename + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        orig_shape = img.shape
        img = cv2.resize(img, (300, 300))
        img -= [123, 117, 104]
        return torch.from_numpy(img).permute(2, 0, 1), orig_shape

    def _get_annotation(self, index, orig_shape):
        h, w, _ = orig_shape
        filename = self.ids[index]
        anno_path = os.path.join(self.anno_dir, filename + '.xml')
        objects = ET.parse(anno_path).findall('object')
        boxes = []
        labels = []

        for obj in objects:
            name = obj.find('name').text.lower().strip()
            bndbox = obj.find('bndbox')
            xmin = (float(bndbox.find('xmin').text) - 1) / w
            ymin = (float(bndbox.find('ymin').text) - 1) / h
            xmax = (float(bndbox.find('xmax').text) - 1) / w
            ymax = (float(bndbox.find('ymax').text) - 1) / h
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(self.name_to_idx[name] + 1)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __getitem__(self, index):
        img, orig_shape = self._get_image(index)
        boxes, labels = self._get_annotation(index, orig_shape)
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)

        return img, boxes, labels


def create_dataloader(root_dir, batch_size, num_examples=-1):
    dataset = VOCDataset('./data/VOCdevkit', '2007', num_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == '__main__':
    voc = VOCDataset('./data/VOCdevkit', '2007')
    data = voc[0]

    idx_to_name = voc.idx_to_name

    print([x.shape for x in data])

    img, boxes, labels = data
    img = img.permute(1, 2, 0).contiguous().numpy()
    img += [123, 117, 104]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes = boxes.numpy() * 300
    labels = labels.numpy()

    visualizer = ImageVisualizer(idx_to_name)
    visualizer.save_image(img, boxes, labels, 'test.jpg')
