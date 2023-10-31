import torch
import numpy as np


def postprocess(prediction, anchors, img_dim):
    # prediction: [1, 3, 13, 13, 85]
    # targets: [10, 5]
    n_batch, n_anchor, H, W, n_dim = prediction.size()
    stride = img_dim[0] / H # 416 / W = 416 / 13 = 32
    device = prediction.device

    x = torch.sigmoid(prediction[..., 0]) # center x: [1, 3, 13, 13]
    y = torch.sigmoid(prediction[..., 1]) # center y: [1, 3, 13, 13]
    w = prediction[..., 2] # width: [1, 3, 13, 13]
    h = prediction[..., 3] # height: [1, 3, 13, 13]
    pred_conf = torch.sigmoid(prediction[..., 4]) # [1, 3, 13, 13]

    # grid_x的shape为[1,1,nG,nG], 每一行的元素为:[0,1,2,3,...,nG-1]
    grid_x = torch.arange(W).repeat(H, 1).view([1, 1, H, W]).to(device)
    # grid_y的shape为[1,1,nG,nG], 每一列元素为: [0,1,2,3, ...,nG-1]
    grid_y = torch.arange(H).repeat(W, 1).t().view(1, 1, H, W).to(device)

    # scaled_anchors 是将原图上的 box 大小根据当前特征图谱的大小转换成相应的特征图谱上的 box, shape: [3, 2]
    scaled_anchors = torch.tensor([(a_h / stride, a_w / stride) for a_h, a_w in anchors]).to(device)

    # 分别获取其 w 和 h, 并将shape形状变为: [1,3,1,1]
    anchor_h = scaled_anchors[:, 0:1].view((1, anchors.size(0), 1, 1))
    anchor_w = scaled_anchors[:, 1:2].view((1, anchors.size(0), 1, 1))
    # shape: [1, 3, 13, 13, 4], 给 anchors 添加 offset 和 scale
    pred_bboxes = torch.stack(
        [x+grid_x, y+grid_y, torch.exp(w)*anchor_w, torch.exp(h)*anchor_h], dim=-1
    ) * stride  # 这里对参数直接乘倍数，恢复正常

    # 非训练阶段则直接返回预测结果, output的shape为: [n_batch, -1, 85]
    output = torch.cat(
        [pred_bboxes.view(n_batch, -1, 4), pred_conf.view(n_batch, -1, 1)], dim=-1
    )
    return output


def nms_single_class(prediction, conf_thres=0.5, nms_thres=0.4):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    移除那些置信度低于conf_thres的boxes，在剩余的boxes上执行NMS算法。
    先选出具有最大score的box，删除与该box交并比大于阈值的box，接着继续选下一个最大socre的box, 重复上述操作，直至bbox为空。

    Arguments: 
        prediction: shape = (B, 2400, 5), 2400是feature map上anchor box的总数。
    Returns:
        output: shape = (B, N, 5)，N是每张图片剩余的bbox
    """
    # xywh->xyxy
    box_corner = np.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    box_corner[:, :, 4] = prediction[:, :, 4]

    output = [None for _ in range(len(box_corner))]
    for image_i, image_pred in enumerate(box_corner):
        # 先清除所有置信度小于conf_thres的box
        detections = image_pred[image_pred[:, 4] >= conf_thres]
        if not detections.shape[0]:
            continue

        # 按照每个box的置信度进行排序(第5维代表置信度 score)
        conf_sort_index = np.argsort(-detections[:, 4])
        detections = detections[conf_sort_index]
        max_detections = []
        while detections.shape[0]:
            # 将具有最大score的box添加到max_detections列表中,
            max_detections.append(detections[0])
            # 当只剩下一个box时, 当前类的nms过程终止
            if len(detections) == 1:
                break
            # 获取当前最大socre的box与其余同类box的iou, 调用了本文件的bbox_iou()函数
            ious = bbox_iou_numpy(max_detections[-1], detections[1:])
            # 移除那些交并比大于阈值的box(也即只保留交并比小于阈值的box)
            detections = detections[1:][ious < nms_thres]
        # 将执行nms后的剩余的同类box连接起来, 最终shape为[m, 5], m为nms后同类box的数量
        max_detections = np.stack(max_detections)
        # 将计算结果添加到output返回值当中, output是一个列表, 列表中的每个元素代表这一张图片的nms后的box
        output[image_i] = max_detections 
    return output


def bbox_iou_numpy(rect1, rectangles, x1y1x2y2=True):
    """
    Arguments:
        rect1: pred bbox, shape=(4,)
        rectangles: target bboxes, shape=(B,4) or (4,)
    Returns:
        iou: iou between rect1 and each rect in rectangles.
    """
    if not x1y1x2y2:
        # 获取 box1 和 box2 的左上角和右下角坐标
        rect1 = np.concatenate([rect1[:2]-rect1[2:]/2, rect1[:2]+rect1[2:]/2])
        rectangles = np.concatenate(
            [rectangles[:, :2] - rectangles[:, 2:] / 2,
             rectangles[:, :2] + rectangles[:, 2:] / 2], axis=-1
        )

    # 计算交集区域的左上角坐标
    x_intersection = np.maximum(rect1[0], rectangles[:, 0])
    y_intersection = np.maximum(rect1[1], rectangles[:, 1])
    
    # 计算交集区域的右下角坐标
    x_intersection_end = np.minimum(rect1[2], rectangles[:, 2])
    y_intersection_end = np.minimum(rect1[3], rectangles[:, 3])
    
    # 计算交集区域的宽度和高度（可能为负数，表示没有重叠）
    intersection_width = np.maximum(0, x_intersection_end - x_intersection)
    intersection_height = np.maximum(0, y_intersection_end - y_intersection)
    
    # 计算交集区域的面积
    intersection_area = intersection_width * intersection_height
    
    # 计算矩形1的面积
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    
    # 计算其他矩形的面积
    area_rectangles = (rectangles[:, 2] - rectangles[:, 0]) * (rectangles[:, 3] - rectangles[:, 1])
    
    # 计算并集区域的面积
    iou = intersection_area / (area_rect1 + area_rectangles - intersection_area + 1e-16)
    return iou
