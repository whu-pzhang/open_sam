from typing import List, Union, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

import torch.nn.functional as F
from open_sam.datasets.sam_data_sample import SampleList

MIN_AREA = 100


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image,
                                    boxes,
                                    colors=['red'] * len(boxes),
                                    labels=labels,
                                    width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image,
                                        masks=masks,
                                        colors=['cyan'] * len(masks),
                                        alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def get_contours(mask):
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    mask = mask.astype(np.uint8)
    mask *= 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    effContours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            effContours.append(c)
    return effContours


def contour_to_points(contour):
    pointsNum = len(contour)
    contour = contour.reshape(pointsNum, -1).astype(np.float32)
    points = [point.tolist() for point in contour]
    return points


def generate_labelme_json(binary_masks, labels, image_size, image_path=None):
    """Generate a LabelMe format JSON file from binary mask tensor.

    Args:
        binary_masks: Binary mask tensor of shape [N, H, W].
        labels: List of labels for each mask.
        image_size: Tuple of (height, width) for the image size.
        image_path: Path to the image file (optional).

    Returns:
        A dictionary representing the LabelMe JSON file.
    """
    num_masks = binary_masks.shape[0]
    binary_masks = binary_masks.numpy()

    json_dict = {
        "version": "4.5.6",
        "imageHeight": image_size[0],
        "imageWidth": image_size[1],
        "imagePath": image_path,
        "flags": {},
        "shapes": [],
        "imageData": None
    }

    # Loop through the masks and add them to the JSON dictionary
    for i in range(num_masks):
        mask = binary_masks[i]
        label = labels[i]
        effContours = get_contours(mask)

        for effContour in effContours:
            points = contour_to_points(effContour)
            shape_dict = {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": points,
                "shape_type": "polygon"
            }

            json_dict["shapes"].append(shape_dict)

    return json_dict


def stack_batch(inputs: dict[List[torch.Tensor]],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                pad_val: Union[int, float] = 0,
                mask_pad_val: Union[int, float] = 255):
    """Stack multiple inputs to form a batch and pad the images and gt_masks
    to the max shape use the right bottom padding mode.

    Args:
        inputs (Dict[List[Tensor]]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SamDataSample`]): The list of data samples.
            It usually includes information such as `gt_masks`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        mask_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    padded_inputs = dict()
    padded_samples = []

    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs['image']]
    max_size = np.stack(inputs_sizes).max(0)

    padded_imgs = []
    images = inputs.pop('image')
    for i in range(len(images)):
        tensor = images[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_imgs.append(pad_img)

        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None

            if 'gt_masks' in data_sample:
                gt_masks = data_sample.gt_instances.data
                del data_sample.gt_instances.data
                data_sample.gt_instances.data = F.pad(gt_masks.unsqueeze(1),
                                                      padding_size,
                                                      value=mask_pad_val)
                pad_shape = data_sample.gt_instances.shape

            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': pad_shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(img_padding_size=padding_size,
                     pad_shape=pad_img.shape[-2:]))

    padded_inputs['image'] = torch.stack(padded_imgs, dim=0)
    for k, v in inputs.items():
        v = torch.stack(v, dim=0)
        # merge image_batch and prompt_batch
        padded_inputs[k] = v.reshape(-1, *v.shape[2:])

    return padded_inputs, padded_samples
