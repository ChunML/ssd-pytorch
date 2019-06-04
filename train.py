import argparse
import os
import torch
from torch import optim
from data import create_dataloader
from network import create_ssd
from losses import create_loss
from anchor import generate_default_boxes

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/VOCdevkit')
parser.add_argument('--pretrained_path', default='./weights/vgg16_reducedfc.pth')
parser.add_argument('--neg_ratio', default=3, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--gamma', default=0.1, type=float)

parser.add_argument('--num_examples', default=-1, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_epochs', default=120, type=int)
parser.add_argument('--checkpoint_dir', default='./models')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 21


def train_step(data, net, criterion, optimizer):
    """ train_step function

    Args:
        data: data of one batch
            img: input images (B, 3, 300, 300)
            gt_confs: classification targets (B, num_anchor)
            gt_locs: regression targets (B, num_anchor, 4)
        net: SSD network
        criterion: loss computing object
        optimizer: training optimizer

    Returns:
        loss: total loss
        conf_loss: classification loss
        loc_loss: regression loss
    """
    net.train()
    img, gt_confs, gt_locs = data
    img = img.to(device)
    gt_confs = gt_confs.to(device)
    gt_locs = gt_locs.to(device)

    optimizer.zero_grad()

    confs, locs = net(img)

    conf_loss, loc_loss = criterion(confs, locs, gt_confs, gt_locs)

    loss = conf_loss + loc_loss

    loss.backward()
    optimizer.step()

    return loss, conf_loss, loc_loss


if __name__ == '__main__':
    config = {
        'scales': [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
        'fm_sizes': [38, 19, 10, 5, 3, 1],
        'ratios': [(2,), (2, 3), (2, 3), (2, 3), (2,), (2,)]
    }

    default_boxes = generate_default_boxes(config)

    dataloader, info = create_dataloader(args.data_dir, args.batch_size, default_boxes, args.num_examples)

    ssd = create_ssd(NUM_CLASSES, 'base', args.pretrained_path)
    ssd.to(device)

    criterion = create_loss(args.neg_ratio, NUM_CLASSES)

    optimizer = optim.SGD(ssd.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.num_epochs*0.6), int(args.num_epochs*0.8)], gamma=0.1, last_epoch=-1)

    for epoch in range(args.num_epochs):
        scheduler.step()
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        for i, data in enumerate(dataloader):
            loss, conf_loss, loc_loss = train_step(
                data, ssd, criterion, optimizer)
            loss = loss.item()
            conf_loss = conf_loss.item()
            loc_loss = loc_loss.item()

            avg_loss = (avg_loss * i + loss) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss) / (i + 1)

            if i % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Conf Loss {:.4f} Loc Loss {:.4f}'.format(
                    epoch + 1, i + 1, avg_loss, avg_conf_loss, avg_loc_loss))

        torch.save(
            ssd.state_dict(),
            os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.pth'.format(epoch)))
