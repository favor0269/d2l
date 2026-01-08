import torch
import basic_operation
import matplotlib.pyplot as plt


def multibox_prior(data, sizes, ratios):

    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # there are 5 anchors (1, 0.75) (1, 0.5) (1, 0.25) (2, 0.75) (0.5 0.75)
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # for normalization
    steps_w = 1.0 / in_width  # for normalization
    # torch.arange return [0, 1, ...]
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # center_h： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    # center_w： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    shift_y, shift_x = torch.meshgrid(center_h, center_w)  # generate 2D composition
    # shift_y tensor([
    #               [0.1250, 0.1250, 0.1250, 0.1250],
    #               [0.3750, 0.3750, 0.3750, 0.3750],
    #               [0.6250, 0.6250, 0.6250, 0.6250],
    #               [0.8750, 0.8750, 0.8750, 0.8750]])
    # shift_x tensor([
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750]])

    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(
        -1
    )  # flatten (By memory layout, row-major order)
    # shape:
    # tensor([0.1250, 0.1250, 0.1250, 0.1250, 0.3750, 0.3750, 0.3750, 0.3750, 0.6250,
    #     0.6250, 0.6250, 0.6250, 0.8750, 0.8750, 0.8750, 0.8750])
    # tensor([0.1250, 0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750, 0.1250,
    #     0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750])

    # now the net can train all image, and doesn't depend on resolution

    # now generate h and w for each anchor
    # w_0 = s * sqrt(H * r / W )
    # h_0 = s * sqrt(W / H / r)
    w_0 = torch.cat(
        [
            size_tensor[0] * torch.sqrt(in_height * ratio_tensor[:] / in_width),
            size_tensor[1:] * torch.sqrt(in_height * ratio_tensor[0] / in_width),
        ],
    )
    h_0 = torch.cat(
        [
            size_tensor[0] * torch.sqrt(in_width / ratio_tensor[:] / in_height),
            size_tensor[1:] * torch.sqrt(in_width / ratio_tensor[0] / in_height),
        ]
    )
    print(w_0.shape)
    print(h_0.shape)

    # 1. torch.stack([-w_0, -h_0, w_0, h_0], dim=0)
    # dim = 0
    # [[-w0_1, -w0_2, -w0_3, -w0_4, -w0_5],
    # [-h0_1, -h0_2, -h0_3, -h0_4, -h0_5],
    # [ w0_1,  w0_2,  w0_3,  w0_4,  w0_5],
    # [ h0_1,  h0_2,  h0_3,  h0_4,  h0_5]]

    # 2. T -> [[-w_0, -h_0, w_0, h_0], ...]
    # 3. repeat(in_height * in_width, 1)
    #    dimension 0: repeat (in_height * in_width) times
    #    dimension 1: repeat 1 times (no repetition)
    #       row1
    #       row2
    #       row3
    #       row4
    #       row5 (begin repeat)
    #       row1
    #       row2
    #       ...
    # *but it equals to stack([-w_0, -h_0, w_0, h_0], dim=1) without T
    anchor_manipulations = (
        torch.stack([-w_0, -h_0, w_0, h_0], dim=0).T.repeat(in_height * in_width, 1) / 2
    )
    print(anchor_manipulations.shape)

    out_grid = torch.stack(
        [shift_x, shift_y, shift_x, shift_y], dim=1
    ).repeat_interleave(boxes_per_pixel, dim=0)
    # shape:
    # {[shift_x, shift_y, shift_x, shift_y], ...}
    # interleave:
    # row1
    # row1
    # row1
    # ...
    # row2
    # center * boxes_per_pixel row repetition interleave
    print(out_grid.shape)

    # the output is [[cx - w/2, cy -h/2, cx + w/2, cy + h/2], ...]
    output = out_grid + anchor_manipulations
    # unsqueeze:
    # (num_anchors, 4) -> (1, num_anchors, 4)
    # this is [1, H*W*num_anchors, 4], convenient for training
    return output.unsqueeze(0)


# in python, default param is object, and its default value will be reused
# so if we change it, it will not reset
# so we need _make_list
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ["b", "g", "r", "m", "c"])

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = basic_operation.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)

        if labels and len(labels) > i:
            text_color = "k" if color == "w" else "w"
            axes.text(
                rect.xy[0],
                rect.xy[1],
                labels[i],
                va="center",
                ha="center",
                fontsize=9,
                color=text_color,
                bbox={"facecolor": color, "lw": 0},
            )


def box_iou(boxes1, boxes2):
    # example:
    # boxes1 [1,1,3,3],[0,0,2,4],[1,2,3,4]] (3, 4)
    # boxes2 [[0,0,3,3],[2,0,5,2]] (2, 4)
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # areas1 = [4, 8, 4] shape: (3,)
    # areas2 = [9, 6] shape: (2,)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # boxes1[:, None, :2] = (3,1,2)
    # [
    #     [[1,1]],  # box1
    #     [[0,0]],  # box2
    #     [[1,2]]   # box3
    # ]
    # boxes2[:, :2] = (2,2)
    # [[0,0],
    # [2,0]]
    # Broadcast mechanism: from last dimension
    # first, 2 == 2
    # second, 1 vs 2, 1 expand to 2
    # third, 3 vs 1, 1 expand to 3
    # box1 (3, 2, 2)
    # [
    #     [[1,1], [1,1]],  # box1 与 boxes2 的每个组合
    #     [[0,0], [0,0]],  # box2 与 boxes2
    #     [[1,2], [1,2]]   # box3 与 boxes2
    # ]
    # box2 (3, 2, 2)
    # [
    #     [[0,0], [2,0]],  # 广播给 box1
    #     [[0,0], [2,0]],  # 广播给 box2
    #     [[0,0], [2,0]]   # 广播给 box3
    # ]
    # final result:
    # inter_upperlefts = torch.tensor([
    #     [[1,1], [2,1]],
    #     [[0,0], [2,0]],
    #     [[1,2], [2,2]]
    # ])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # result:
    # inter_lowerrights = torch.tensor([
    #     [[3,3], [3,2]],
    #     [[2,3], [2,2]],
    #     [[3,3], [3,2]]
    # ])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # result:
    # inters = torch.tensor([
    #     [[2,2], [1,1]],
    #     [[2,3], [0,2]],
    #     [[2,1], [1,0]]
    # ])
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # result:
    # inter_areas = torch.tensor([
    #     [4, 1],
    #     [6, 0],
    #     [2, 0]
    # ])

    # area 1 [:, None] (3, 1)
    # [[4],[8],[4]]
    # area 2 (2)
    # [[9, 6]]
    # broadcast:
    # 1 vs 2 -> [[4, 4], [8, 8], [4, 4]]
    # 3 vs 1 -> [[9, 6], [9, 6], [9, 6]]
    union_areas = areas1[:, None] + areas2 - inter_areas
    # return the iou
    return inter_areas / union_areas


# example:
# ground_truth = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.55, 0.2, 0.9, 0.88]])
# anchors = torch.tensor(
#     [
#         [0, 0.1, 0.2, 0.3],
#         [0.15, 0.2, 0.4, 0.4],
#         [0.63, 0.05, 0.88, 0.98],
#         [0.66, 0.45, 0.8, 0.8],
#         [0.57, 0.3, 0.92, 0.9],
#     ]
# )


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)

    # tensor([[0.0536, 0.0000],
    #         [0.1417, 0.0000],
    #         [0.0000, 0.5657],
    #         [0.0000, 0.2059],
    #         [0.0000, 0.7459]])

    # map anchor to gt
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    max_ious, indices = torch.max(jaccard, dim=1)
    # result:
    # max_ious = [0.0536, 0.1417, 0.5657, 0.2059, 0.7459]
    # indices = [0, 0, 1, 1, 1]
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    # r: anc_i = [2, 4]
    box_j = indices[max_ious >= 0.5]
    # r: box_j = [1, 1]

    # let renew the map:
    anchors_bbox_map[anc_i] = box_j

    # Greedy algorithm:
    col_discard = torch.full((num_anchors,), -1)  # [-1, -1, -1, -1, -1]
    row_discard = torch.full((num_gt_boxes,), -1)  # [-1, -1]

    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        # transform to 2D index
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        # bond them strictly
        anchors_bbox_map[anc_idx] = box_idx
        # delete this line and row in jaccard
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    # here map:
    # 1->0 2->1 4->1
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = basic_operation.box_corner_to_center(anchors)
    c_assigned_bb = basic_operation.box_corner_to_center(assigned_bb)
    # 10 and 5 are constant value
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        # shared anchor, own gt
        label = labels[i, :, :]
        # drop the class_id so 1:
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)

        # repeat(1, 4): to match x y w h (all detla) to compute loss
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


plt.figure(figsize=(3.5, 2.5))
sizes = [0.75, 0.5, 0.25]
ratios = [1, 2, 0.5]  # w/h

# Create a simple test image or load from a valid path
# If you have a real image, replace this path
try:
    img = plt.imread("/home/favor0269/pytorch/img/catdog.jpg")
except FileNotFoundError:
    print("Image not found. Creating a random test image.")
    img = torch.rand(300, 400, 3).numpy()

fig = plt.imshow(img)
plt.axis("on")

# use the actual image size to construct a dummy feature map
in_height, in_width = img.shape[:2]
data = torch.zeros((1, 3, in_height, in_width))

# Generate anchors using multibox_prior based on the image grid
boxes = multibox_prior(data, sizes, ratios)
print(f"boxes shape: {boxes.shape}")

device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
# there are 5 anchors (1, 0.75) (1, 0.5) (1, 0.25) (2, 0.75) (0.5 0.75)
boxes_per_pixel = num_sizes + num_ratios - 1

# boxes has shape (1, in_height*in_width*boxes_per_pixel, 4)
# reshape using feature map size, NOT image size
boxes_show = boxes.reshape(1, in_height, in_width, boxes_per_pixel, 4)
bbox_scale = torch.tensor((in_width, in_height, in_width, in_height))

center_h = in_height // 2  # // means 1./; 2.floor result
center_w = in_width // 2
show_bboxes(
    fig.axes,
    boxes_show[0, center_h, center_w, :, :] * bbox_scale,
    ["s=0.75, r=1", "s=0.5, r=1", "s=0.25, r=1", "s=0.75, r=2", "s=0.75, r=0.5"],
)
plt.show()
