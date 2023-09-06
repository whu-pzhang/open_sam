import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from mmengine.runner import Runner

from torchvision.transforms import Compose

from open_sam.datasets.whu_building import WHUBuildingDataset
from open_sam.sam_inferencer import SAMInferencer
from open_sam.modeling.sam import SAM
from open_sam.datasets.transforms import ResizeLongestSide, PackSamInputs
from open_sam.datasets.utils import custom_collate_fn
from open_sam.datasets import SegDataset

from mmseg.registry import MODELS, DATASETS, TRANSFORMS
from mmseg.utils import register_all_modules

register_all_modules()


def sam_batch_predict(predictor, input_bboxes, img_hw, batch_size=32):
    num_bboxes = len(input_bboxes)
    num_batches = int(np.ceil(num_bboxes / batch_size))

    masks = []
    input_bboxes_tensor = torch.from_numpy(input_bboxes).to(predictor.device)
    for i in range(num_batches):
        left_index = i * batch_size
        right_index = (i + 1) * batch_size
        if i == num_batches - 1:
            batch_boxes = input_bboxes_tensor[left_index:]
        else:
            batch_boxes = input_bboxes_tensor[left_index:right_index]

        transformed_boxes = predictor.transform.apply_boxes_torch(
            batch_boxes, img_hw)

        batch_masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            return_logits=False)

        batch_masks = batch_masks.squeeze(1)
        masks.extend([*batch_masks])

    masks = torch.stack(masks, dim=0)
    return masks


def show_mask(mask, ax, random_color=False, alpha=0.8):
    if random_color:
        color = np.concatenate(
            [np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0],
               pos_points[:, 1],
               color='green',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='yellow',
                      facecolor=(0, 0, 0, 0),
                      lw=2))


def show_data_sample(sample, figsize=(8, 8)):
    print(sample['img_path'])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(sample['img'])

    for idx, mask in enumerate(sample['gt_masks']):
        show_mask(mask, ax, random_color=True)
    for box in sample['boxes']:
        show_box(box, ax)

    point_coords = sample['point_coords']  # BxNx2
    point_labels = np.ones(point_coords.shape[:2], dtype=np.uint8)  # BxN
    show_points(point_coords, point_labels, ax, marker_size=200)

    # ax.axis('off')
    plt.tight_layout()
    plt.show()


def test_dataset():
    dataset_cfg = dict(
        type='WHUBuildingDataset',
        data_root='data/whu-building/cropped_aerial_data',
        split='train',
        ann_file='data/whu-building/train.txt',
        max_objects=15,
        points_per_instance=2,
        transform=ResizeLongestSide(1024),
    )
    ds = DATASETS.build(dataset_cfg)

    sample = ds[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(sample['img'])
    for idx, mask in enumerate(sample['gt_masks']):
        show_mask(mask, ax, random_color=True)
    for box in sample['boxes']:
        show_box(box, ax)

    point_coords = sample['point_coords']  # BxNx2
    point_labels = np.ones(point_coords.shape[:2], dtype=np.uint8)  # BxN
    show_points(point_coords, point_labels, ax, marker_size=200)

    # ax.axis('off')
    plt.tight_layout()
    plt.show()

    train_dataloader = dict(dataset=ds,
                            batch_size=4,
                            num_workers=4,
                            sampler=dict(type='DefaultSampler', shuffle=True),
                            drop_last=True,
                            collate_fn=dict(type='custom_collate_fn'))

    loader = Runner.build_dataloader(train_dataloader)

    for idx, data_batch in enumerate(loader):
        if idx == 0:
            print(data_batch.keys())
        print(f'{idx+1}/{len(loader)}: ', data_batch['img_shape'])
        # for box in data_batch['boxes']:
        #     print(box.shape)
        break


def test_sam_predict():
    dataset_cfg = dict(type='WHUBuildingDataset',
                       data_root='data/whu-building/cropped_aerial_data',
                       split='train',
                       ann_file='data/whu-building/train.txt',
                       max_objects=15,
                       points_per_instance=2)
    ds = DATASETS.build(dataset_cfg)

    sample = ds[1]
    print(sample['filename'])
    print(sample['gt_masks'].shape)

    def sam_infer(image,
                  prompt_boxes,
                  point_coords,
                  point_labels,
                  multimask_output=False):
        ori_size = image.shape[:2]

        predictor = SAMInferencer(arch='base')
        sam = predictor.model
        device = predictor.device
        transform = ResizeLongestSide(1024)

        input_image = transform.apply_image(image)
        input_image = torch.as_tensor(input_image, device=device)
        input_image = input_image.permute(2, 0, 1).contiguous()

        # boxes = torch.as_tensor(prompt_boxes).to(device)
        # boxes = transform.apply_boxes_torch(boxes, ori_size)
        boxes = transform.apply_boxes(prompt_boxes, ori_size)
        boxes = torch.as_tensor(boxes).to(device)

        # point_coords = torch.as_tensor(point_coords).to(device)
        # point_coords = transform.apply_coords_torch(point_coords, ori_size)
        point_coords = transform.apply_coords(point_coords, ori_size)
        point_coords = torch.as_tensor(point_coords).to(device)
        point_labels = torch.as_tensor(point_labels).to(device)

        batch_input = [
            dict(
                image=input_image,
                original_size=ori_size,
                boxes=boxes,
                point_coords=point_coords,
                point_labels=point_labels,
            )
        ]

        output = sam(batch_input,
                     data_samples=None,
                     mode='predict',
                     multimask_output=multimask_output)

        return output

    image = sample['img']  # 3XHxW
    boxes = sample['boxes']  # Bx4
    point_coords = sample['point_coords']  # BxNx2
    point_labels = np.ones(point_coords.shape[:2], dtype=np.uint8)  # BxN

    # inference
    batched_output = sam_infer(image,
                               boxes,
                               point_coords,
                               point_labels,
                               multimask_output=True)

    print(batched_output[0].keys())
    print(batched_output[0]['masks'].shape)
    iou_predictions = batched_output[0]['iou_predictions']
    scores, mask_idxs = torch.max(iou_predictions, dim=1, keepdim=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(image)
    for idx, mask in enumerate(batched_output[0]['masks']):
        best_mask = mask[mask_idxs[idx]]
        show_mask(best_mask.cpu().numpy(), ax, random_color=True)
    for box in boxes:
        show_box(box, ax)
    show_points(point_coords, point_labels, ax, marker_size=200)

    # ax.axis('off')
    plt.tight_layout()
    plt.show()


def test_sam_loss():
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', imdecode_backend='pillow'),
        dict(type='RandomResize',
             scale=(2048, 512),
             ratio_range=(0.5, 2.0),
             keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='ResizeLongestEdge', scale=1024),
        dict(type='GenerateSAMPrompt', max_instances=15,
             points_per_instance=2),
        # dict(type='ResizeLongestSide', target_length=1024),
        dict(type='PackSamInputs'),
    ]
    ds = SegDataset(data_root='data/whu-building/cropped_aerial_data',
                    img_suffix='.tif',
                    seg_map_suffix='.tif',
                    data_prefix=dict(img_path='train/image',
                                     seg_map_path='train/label'),
                    ann_file='../train.txt',
                    pipeline=pipeline)

    train_dataloader = dict(
        dataset=ds,
        batch_size=2,
        num_workers=0,
        sampler=dict(type='DefaultSampler', shuffle=True),
        drop_last=True,
    )

    loader = Runner.build_dataloader(train_dataloader)

    def sam_train(model, loader):
        device = model.device
        for idx, data_batch in enumerate(loader):

            inputs = {
                k: [i.to(device) for i in v]
                for k, v in data_batch['inputs'].items()
            }
            data_samples = [
                item.to(device) for item in data_batch['data_samples']
            ]

            losses, logits = model(inputs, data_samples, mode='loss')

            print(losses)

            for logit, gt_mask in zip(logits, data_samples):
                gt_mask = gt_mask.gt_instances.masks.data.detach().cpu()

                print(logit.shape, gt_mask.shape)

                f, ax = plt.subplots(1, 2)
                for m in gt_mask.numpy():
                    show_mask(m, ax[0], random_color=True)

                for m in logit.squeeze():
                    m = (torch.sigmoid(m) > 0.5).numpy()
                    show_mask(m, ax[1], random_color=True)

                plt.tight_layout()
                plt.show()

            if idx == 5:
                break

        # ori_size = image.shape[:2]

        # predictor = SAMInferencer(arch='base')
        # sam = predictor.model
        # device = predictor.device
        # transform = ResizeLongestSide(1024)

        # input_image = transform.apply_image(image)
        # input_image = torch.as_tensor(input_image, device=device)
        # input_image = input_image.permute(2, 0, 1).contiguous()

        # boxes = torch.as_tensor(prompt_boxes).to(device)
        # boxes = transform.apply_boxes_torch(boxes, ori_size)

        # point_coords = torch.as_tensor(point_coords).to(device)
        # point_coords = transform.apply_coords_torch(point_coords, ori_size)
        # point_labels = torch.as_tensor(point_labels).to(device)

        # batch_input = [
        #     dict(
        #         image=input_image,
        #         original_size=ori_size,
        #         boxes=boxes,
        #         point_coords=point_coords,
        #         point_labels=point_labels,
        #     )
        # ]

        # data_samples = [
        #     dict(gt_masks=gt_masks, img_path=None, mask_path=None),
        # ]

        # output = sam(batch_input, data_samples=data_samples, mode='loss')

        # return output

    # image = sample['image']  # 3XHxW
    # boxes = sample['boxes']  # Bx4
    # point_coords = sample['point_coords']  # BxNx2
    # point_labels = np.ones(point_coords.shape[:2], dtype=np.uint8)  # BxN
    # gt_masks = sample['gt_masks']  # BxHxW

    # # print(gt_masks.shape)

    # losses = sam_infer(image, boxes, point_coords, point_labels, gt_masks)
    # print(losses)

    sam = build_sam(arch='base')

    for p in sam.image_encoder.parameters():
        p.requires_grad = False

    sam_train(sam, loader)


def test_base_dataset():
    from open_sam.datasets.base import SegDataset

    register_all_modules()

    def vis_data():
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', imdecode_backend='pillow'),
            dict(type='RandomResize',
                 scale=(2048, 512),
                 ratio_range=(0.5, 2.0),
                 keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='ResizeLongestEdge', scale=1024),
            dict(type='GenerateSAMPrompt',
                 max_instances=15,
                 points_per_instance=1),
            # dict(type='ResizeLongestSide', target_length=1024),
            # dict(type='PackSamInputs'),
        ]
        ds = SegDataset(data_root='data/whu-building/cropped_aerial_data',
                        img_suffix='.tif',
                        seg_map_suffix='.tif',
                        data_prefix=dict(img_path='train/image',
                                         seg_map_path='train/label'),
                        ann_file='../train.txt',
                        pipeline=pipeline)
        # ds = SegDataset(data_root='data/iSAID',
        #                 img_suffix='.png',
        #                 seg_map_suffix='_instance_color_RGB.png',
        #                 data_prefix=dict(img_path='img_dir/train',
        #                                  seg_map_path='ann_dir/train'),
        #                 pipeline=pipeline)
        show_data_sample(ds[100])

    def load_data():
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', imdecode_backend='pillow'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='ResizeLongestEdge', scale=1024),
            dict(type='GenerateSAMPrompt',
                 max_instances=15,
                 points_per_instance=2),
            # dict(type='ResizeLongestSide', target_length=1024),
            dict(type='PackSamInputs'),
        ]
        ds = SegDataset(data_root='data/whu-building/cropped_aerial_data',
                        img_suffix='.tif',
                        seg_map_suffix='.tif',
                        data_prefix=dict(img_path='train/image',
                                         seg_map_path='train/label'),
                        ann_file='../train.txt',
                        pipeline=pipeline)
        train_dataloader = dict(
            dataset=ds,
            batch_size=4,
            num_workers=4,
            persistent_workers=True,
            sampler=dict(type='InfiniteSampler', shuffle=True),
            drop_last=True,
        )

        loader = Runner.build_dataloader(train_dataloader)

        for idx, data_batch in enumerate(loader):
            if idx == 0:
                print(data_batch['inputs'].keys())
            print(f'{idx+1}/{len(loader)}: ',
                  data_batch['data_samples'][0].metainfo)
            # for box in data_batch['boxes']:
            #     print(box.shape)
            # break

    # vis_data()
    load_data()


def build_sam(arch):
    from mmengine.runner.checkpoint import load_checkpoint
    from open_sam.sam_inferencer import model_zoo

    cfg = dict(
        type='SAM',
        image_encoder=dict(type='mmpretrain.ViTSAM',
                           arch=arch,
                           img_size=1024,
                           patch_size=16,
                           out_channels=256,
                           use_abs_pos=True,
                           use_rel_pos=True,
                           window_size=14),
        prompt_encoder=dict(
            type='PromptEncoder',
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ),
        mask_decoder=dict(
            type='MaskDecoder',
            num_multimask_outputs=3,
            transformer=dict(
                type='TwoWayTransformer',
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        loss_decode=[
            dict(type='FocalLoss', use_sigmoid=True),
            dict(type='CrossEntropyLoss',
                 use_sigmoid=True,
                 avg_non_ignore=True),
            dict(type='DiceLoss'),
        ],
    )

    model = MODELS.build(cfg)

    load_checkpoint(model, model_zoo.get(arch), strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


if __name__ == '__main__':
    # test_dataset()
    # test_sam_predict()
    test_sam_loss()
    # print(TRANSFORMS)
    # test_base_dataset()
