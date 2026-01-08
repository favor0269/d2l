import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), dim=-1)
    return boxes


def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes

def set_fig():
    plt.figure(figsize=(3.5, 2.5))
    img = plt.imread("/home/favor0269/pytorch/img/catdog.jpg")
    plt.imshow(img)
    plt.axis("on")


dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]


def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False,
        edgecolor=color,
        linewidth=2,
    )


if __name__ == "__main__":
    ax = plt.gca()
    ax.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    ax.add_patch(bbox_to_rect(cat_bbox, 'red'))
    plt.show()
