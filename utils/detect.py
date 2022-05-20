import os
import cv2
import numpy as np


def get_up_left_coordinates(image, size=(5, 5), stride=(2, 2), padding="VALID"):
    """
    This op gets up-left-corner coordinates of patches in input images.

    Args:
        image: A numpy.ndarray with shape [in_rows, in_cols, depth], the input image
        size: A tuple or list like [size_rows, size_cols], The size of extracted patches;
        stride: A tuple or list like [stride_rows, stride_cols], How far the centers of
            two consecutive patches are in the images;
        padding: String, "SAME" or "VALID"

    Returns:
        up_left_coordinates: A numpy.ndarray with shape [number_rows, number_cols, 2],
            e.x. [[[y1, x1]],[[y2, x2]]]

    """
    image_shape = image.shape

    if padding.upper() == "VALID":
        number_rows = (image_shape[0] - size[0]) // stride[0] + 1
        number_cols = (image_shape[1] - size[1]) // stride[1] + 1
    elif padding.upper() == "SAME":
        number_rows = image_shape[0] // stride[0] + 1
        number_cols = image_shape[1] // stride[1] + 1
    else:
        raise ValueError("padding MUST be 'SAME' or 'VALID'")

    rows_coordinates = np.arange(number_rows)
    cols_coordinates = np.arange(number_cols)
    rows_coordinates, cols_coordinates = np.meshgrid(rows_coordinates, cols_coordinates)

    up_left_coordinates = np.stack((rows_coordinates, cols_coordinates), axis=-1)
    up_left_coordinates = np.transpose(up_left_coordinates, (1, 0, 2))
    up_left_coordinates = up_left_coordinates * stride

    return up_left_coordinates


def pad_image(image, size=(5, 5), stride=(2, 2), padding="VALID"):
    """
    This op pads input images.

    Args:
        image: A numpy.ndarray with shape [in_rows, in_cols, depth], the input image
        size: A tuple or list like [size_rows, size_cols], The size of extracted patches;
        stride: A tuple or list like [stride_rows, stride_cols], How far the centers of
            two consecutive patches are in the images;
        padding: String, "SAME" or "VALID"

    Returns:
        padding_image: numpy.ndarray, the images after padding
    """
    padding_image = image
    image_shape = image.shape

    if padding.upper() == "VALID":
        pass
    elif padding.upper() == "SAME":
        rows_padding_pixel = image_shape[0] // stride[0] * stride[0] + size[0] - image_shape[0]
        rows_padding_pixel = rows_padding_pixel if rows_padding_pixel > 0 else 0
        cols_padding_pixel = image_shape[1] // stride[1] * stride[1] + size[1] - image_shape[1]
        cols_padding_pixel = cols_padding_pixel if cols_padding_pixel > 0 else 0

        padding_image = np.pad(
            padding_image,
            ((0, rows_padding_pixel), (0, cols_padding_pixel), (0, 0)),
            'constant',
            constant_values=(0, 0)
        )
    else:
        raise ValueError("padding MUST be 'SAME' or 'VALID'")

    return padding_image


def extract_image_patches(image, image_path=None, size=(10, 10), stride=(10, 10), rate=(1, 1), padding="VALID"):
    """
    This op collects patches from the input image, as if applying a convolution.
    All extracted patches are stacked in the depth (last) dimension of the output.
    Specifically, the op extracts patches of shape sizes which are strides apart
    in the input image. The output is subsampled using the rates argument, in the
    same manner as "atrous" or "dilated" convolutions.

    Args:
        image: A numpy.ndarray with shape [in_rows, in_cols, depth];
        image_path: A String, image file path
        size: A tuple or list like [size_rows, size_cols], The size of extracted patches;
        stride: A tuple or list like [stride_rows, stride_cols], How far the centers of
            two consecutive patches are in the images;
        rate: A tuple or list like [rate_rows, rate_cols]. This is the input stride,
            specifying how far two consecutive patch samples are in the input. Equivalent
            to extracting patches with
            "patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)", followed
            by subsampling them spatially by a factor of rates. This is equivalent to rate in
            dilated (a.k.a. Atrous) convolutions;
        padding: String, "SAME" or "VALID". The type of padding algorithm to use.The padding
            argument has no effect on the size of each patch, it determines how many patches
            are extracted. If VALID, only patches which are fully contained in the input image
            are included. If SAME, all patches whose starting point is inside the input are
            included, and areas outside the input default to zero.

    Returns:
        up_left_coordinates: A numpy.ndarray with shape [number_patches, 2], e.x. [[y1, x1],[y2, x2]]
        patches: A numpy.ndarray with shape [number_patches, size_rows, size_cols, depth];
    """

    assert len(size) == 2, "The length of size is not 2"
    assert len(stride) == 2, "The length of stride is not 2"
    assert len(rate) == 2, "The length of rate is not 2"
    assert padding.upper() in ["SAME", "VALID"], "padding MUST be 'SAME' or 'VALID'"

    if image is None and image_path:
        image = cv2.imread(image_path)
    else:
        raise ValueError("Both Image data and filepath are None")

    size = (
        size[0] + (size[0] - 1) * (rate[0] - 1),
        size[1] + (size[1] - 1) * (rate[1] - 1),
    )  # update patch_size

    up_left_coordinates = get_up_left_coordinates(image, size, stride, padding)
    image_padding = pad_image(image, size, stride, padding)

    patches = []
    for y, x in np.reshape(up_left_coordinates, (-1, 2)):
        patches.append(
            image_padding[y:y + size[0], x:x + size[1]],
        )
    patches = np.stack(patches, axis=0)
    patches = np.reshape(patches, list(up_left_coordinates.shape[:-1]) + [-1])

    return image, up_left_coordinates, patches


def fuse_results(result_dir_path, size, image_size):
    assert os.path.exists(result_dir_path), "PathError: {} DOES NOT exist!"
    files = [os.path.join(root, file) for root, dirs, files in os.walk(result_dir_path)
             for file in files if os.path.splitext(file)[1] == '.txt']

    # 解析标签
    original_height, original_width = image_size[:2]
    clip_height, clip_width = size[:2]
    labels = []

    for file in files:

        # 根据文件名解析图片左上角点在原图中的坐标
        file_name = os.path.split(file)[1]  # e.g. 400_600.text
        file_name = os.path.splitext(file_name)[0]  # e.g. 400_600
        y_lu, x_lu = list(map(int, file_name.split('_')))  # y=400, x=600

        with open(file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 1:
            continue

        for line in lines:
            # 获取小图中结果信息
            class_id, x_center, y_center, width, height = list(
                map(
                    float,
                    line.strip('\n').split())
            )

            # 将小图结果迁移到原始图像上
            class_id = int(class_id)
            x_center = (x_center * clip_width + x_lu) / original_width
            y_center = (y_center * clip_height + y_lu) / original_height
            width = width * clip_width / original_width
            height = height * clip_height / original_height

            # 追加到标签列表
            labels.append([class_id, x_center, y_center, width, height])

    return labels


# 过滤labels
def filter_labels(labels, area_threshold=0, img_shape=(1, 1)):
    """
    This op filters labels by the bounding box area

    Args:
        labels: A 'numpy.ndarray' with shape [in_rows, [class_id，x_center, y_center, width, height]],
            the YOLO output labels, e.g.
            [
                [3, 0.809211, 0.928947, 0.381579, 0.142105]
                [3, 0.0723684, 0.306579, 0.144737, 0.465789]
                [3, 0.772368, 0.177632, 0.455263, 0.35]
            ]
        area_threshold: float, if area_threshold > 1, then the threshold means the minimum of bounding
            boxes pixels; if 0 <= area_threshold <= 1, then the threshold means the ratio of minimum to
            image shape

        img_shape: A tuple or list, the shape of image, e.g. (10, 15) means that the height is 10 and
            width 15

    Returns:
        labels: A numpy.ndarray with shape [out_rows, [class_id，x_center, y_center, width, height]]

    """
    assert area_threshold >= 0, "The area threshold MUST NOT be less than 0!"
    assert area_threshold < img_shape[0] * img_shape[1], "The image area must be over area threshold!"

    height, width = img_shape
    area_threshold = area_threshold if area_threshold > 1 else area_threshold * height * width

    boxes = convert_xywh_to_xyxy(labels[..., 1:])
    boxes[..., [0, 2]] = boxes[..., [0, 2]] * width
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, width - 1)

    boxes[..., [1, 3]] = boxes[..., [1, 3]] * height
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, height - 1)

    # print(f'width is {width}')
    # print(f'height is {height}')
    # print(f'boxes is {boxes}')

    boxes = convert_xyxy_to_xywh(boxes)
    boxes_area = boxes[..., 2] * boxes[..., 3]
    return labels[np.where(boxes_area > area_threshold)]


# 融合label
def fuse_labels(labels, iou_threshold=0.9):
    """
    This op fuses labels by iou, if two bonding boxes IOU is over threshold, the op fuses
    the two boxes because the two detected the same objects

    Args:
        labels: A 'numpy.ndarray' with shape [in_rows, [class_id，x_center, y_center, width, height]],
            the YOLO output labels, e.g.
            [
                [3, 0.809211, 0.928947, 0.381579, 0.142105]
                [3, 0.0723684, 0.306579, 0.144737, 0.465789]
                [3, 0.772368, 0.177632, 0.455263, 0.35]
            ]
        iou_threshold: float, if two boxes IOU is over threshold, the two detected the same object

    Returns:
        labels: A numpy.ndarray with shape [out_rows, [class_id，x_center, y_center, width, height]], Any two
            boxes IOU is below threshold

    """
    index = 0  # 最多执行 10 轮融合
    while index < 10:
        boxes = labels[..., 1:]
        #         print(index)

        # 计算 IOU
        iou_matrix = compute_iou(boxes, boxes)

        # 清除对角线的 IOU 指，自身和自身的IOU没意义
        diagonal_mask = np.logical_not(
            np.eye(
                boxes.shape[0]
            ).astype('bool')
        )
        iou_matrix = np.where(diagonal_mask, iou_matrix, 0.0)

        # 寻找各个 box 最大的 IOU，以及对应的 box 指针
        max_iou = np.max(iou_matrix, axis=1)
        matched_box_index = np.argmax(iou_matrix, axis=1)
        matched_box = boxes[matched_box_index]

        # 只有 IOU 大于阈值才合并
        positive_mask = np.greater_equal(max_iou, iou_threshold)
        #         print(max_iou)

        # 只要还有需要融合的区域就继续，否则返回
        if np.any(positive_mask):
            # 融合后的 box 只保留面积大的box
            ignore_mask = np.less_equal(
                boxes[..., 2] * boxes[..., 3],
                matched_box[..., 2] * matched_box[..., 3]
            )

            # 融合不同区域即生成最靠近和最远离图像左上角（原点）的矩形区域
            boxes_fused = np.zeros_like(boxes)
            boxes_fused[..., :2] = np.minimum(
                convert_xywh_to_xyxy(boxes)[..., :2],
                convert_xywh_to_xyxy(matched_box)[..., :2],
            )
            boxes_fused[..., 2:] = np.maximum(
                convert_xywh_to_xyxy(boxes)[..., 2:],
                convert_xywh_to_xyxy(matched_box)[..., 2:],
            )
            boxes_fused = np.clip(boxes_fused, 0.0, 1.0)
            boxes_fused = convert_xyxy_to_xywh(boxes_fused)

            boxes = np.where(positive_mask[..., None], boxes_fused, boxes)
            labels[..., 1:] = boxes

            # 删除已经融合但面积小的box
            labels = labels[
                np.logical_not(
                    np.logical_and(positive_mask, ignore_mask)
                )
            ]
            index += 1
        else:
            break

    return labels


def compute_iou(boxes1, boxes2):
    """
    Computes pairwise IOU matrix for given two sets of boxes

    Args:
        boxes1: A numpy.ndarray with shape `(N, 4)` representing bounding boxes
            where each box is of the format `[x, y, width, height]`.
        boxes2: A numpy.ndarray with shape `(M, 4)` representing bounding boxes
            where each box is of the format `[x, y, width, height]`.

    Returns:
    pairwise IOU matrix with shape `(N, M)`, where the value at ith row jth column
        holds the IOU between ith box and jth box from boxes1 and boxes2 respectively.
    """
    boxes1_xyxy = convert_xywh_to_xyxy(boxes1)
    boxes2_xyxy = convert_xywh_to_xyxy(boxes2)

    # 计算重叠面积
    lu = np.maximum(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])  # 广播
    rd = np.minimum(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])
    intersection = np.maximum(0.0, rd - lu)  # 获得重叠区域宽高
    intersection = intersection[:, :, 0] * intersection[:, :, 1]  # 获得重叠区域面积

    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union = np.maximum(
        boxes1_area[:, None] + boxes2_area - intersection, 1e-5
    )

    return np.clip(intersection / union, 0.0, 1.0)


def save_patches(save_dir, patches, up_left_coordinates, size):
    number_rows, number_cols, elements = patches.shape
    for row in range(number_rows):
        for col in range(number_cols):
            y, x = up_left_coordinates[row, col]
            patch_image = np.reshape(patches[row, col], (size[0], size[1], -1))
            cv2.imwrite(
                os.path.join(save_dir, f'{y}_{x}.png'),
                patch_image.astype("uint8")
            )


def parse_patches_detection(label_dir_path, image, size):
    files = [os.path.join(root, file) for root, dirs, files in os.walk(label_dir_path)
             for file in files if os.path.splitext(file)[1] == '.txt']

    original_height, original_width = image.shape[:2]
    clip_height, clip_width = size[:2]
    labels = []

    for file in files:

        # 根据文件名解析图片左上角点在原图中的坐标
        # 如果解析失败，则文件名不好含左上角信息，使用默认值(0,0)
        try:
            file_name = os.path.split(file)[1]  # e.g. 400_600.txt
            file_name = os.path.splitext(file_name)[0]  # e.g. 400_600
            y_lu, x_lu = list(map(int, file_name.split('_')))  # y=400, x=600
        except Exception:
            y_lu, x_lu = 0, 0

        with open(file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 1:
            continue

        for line in lines:
            # 获取小图中结果信息
            class_id, x_center, y_center, width, height = list(
                map(
                    float,
                    line.strip('\n').split())
            )

            # 将小图结果迁移到原始图像上
            class_id = int(class_id)
            x_center = (x_center * clip_width + x_lu) / original_width
            y_center = (y_center * clip_height + y_lu) / original_height
            width = width * clip_width / original_width
            height = height * clip_height / original_height

            # 追加到标签列表
            labels.append([class_id, x_center, y_center, width, height])

    return np.asarray(labels)


# 工具函数，修建输入值
def clip_value(x, xmin=0, xmax=1e5):
    if x <= xmin:
        x = xmin
    if x >= xmax:
        x = xmax - 1
    return x


def convert_xywh_to_xyxy(boxes):
    """Changes the box format to [xmin, ymin, xmax, ymax].

    Arguments:
      boxes: A numpy.ndarray of rank 2 or higher with a shape of `(num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x_center, y_center, width, height]`.

    Returns:
        converted boxes with shape same as that of boxes.
    """
    return np.concatenate(
        [boxes[...,:2] - 0.5 * boxes[...,2:], boxes[...,:2] + 0.5 * boxes[...,2:]],
        axis=-1,
    )


def convert_xyxy_to_xywh(boxes):
    """Changes the box format to [x_center, y_center, width, height].

    Arguments:
      boxes: A numpy.ndarray of rank 2 or higher with a shape of `(num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
        converted boxes with shape same as that of boxes.
    """
    return np.concatenate(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


