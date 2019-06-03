import os
import cv2


def visualize_and_save_image(img, boxes, labels, idx_to_name, save_path):
    for i, box in enumerate(boxes):
        top_left = (box[0], box[1])
        bot_right = (box[2], box[3])
        cv2.rectangle(
            img, top_left, bot_right,
            color=(0, 255, 0), thickness=2)
        cv2.putText(
            img, idx_to_name[labels[i] - 1],
            top_left,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.7, color=(255, 255, 255))
        cv2.imwrite(save_path, img)
