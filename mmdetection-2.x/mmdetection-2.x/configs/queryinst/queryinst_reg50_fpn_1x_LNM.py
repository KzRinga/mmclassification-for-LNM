# _base_ = [
#     '../_base_/datasets/coco_instance.py', '../_base_/default_runtime.py'
# ]


# dataset settings
dataset_type = 'EC05GSDetDataset'
data_root = '/media/inno/data/gastroscope-det/TrainData/COCO_FORMAT_UPCA_EC05_V54/'
CLASSES = 11
IG_CLASSES = []
size = 544

img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.40, 57.12, 57.38], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=(size, size),
        keep_ratio=False),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='MedianBlur', blur_limit=5, p=1.0),
                    dict(type='MotionBlur', blur_limit=7, p=1.0),
                    dict(type='GaussianBlur', blur_limit=7, p=1.0),
                ],
                p=1.0),
            # dict(type='RandomBrightnessContrast',
            #      brightness_limit=[0.1, 0.3],
            #      contrast_limit=[0.1, 0.3],
            #      p=0.5),
            # dict(type='GaussNoise', p=0.5),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=False),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/train',
        percent=1.0,
        ignore_classes=IG_CLASSES,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/test',
        percent=1.0,
        ignore_classes=IG_CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/test',
        percent=1.0,
        ignore_classes=IG_CLASSES,
        pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'], proposal_nums=(5, 10, 50))
evaluation = dict(metric=['bbox'], proposal_nums=(5, 10, 50))

# 从6依次降低
num_stages = 3
num_proposals = 50

search_best_basis = ['bbox_p2_mAP', 'bbox_p2_recall']
model = dict(
    type='QueryInst',
    backbone=dict(
        type='RegNet',
        arch='regnetx_3.2gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 432, 1008],
        # in_channels=[256, 512, 1024, 2048],
        # out_channels=256,
        out_channels=128,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        # proposal_feature_channel=256),
        proposal_feature_channel=128),
    roi_head=dict(
        type='QueryRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        # proposal_feature_channel=256,
        proposal_feature_channel=128,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            # out_channels=256,
            out_channels=128,
            featmap_strides=[4, 8, 16, 32]),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            # out_channels=256,
            out_channels=128,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=CLASSES,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                # feedforward_channels=2048,
                # in_channels=256,
                feedforward_channels=1024,
                in_channels=128,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    # in_channels=256,
                    # feat_channels=64,
                    # out_channels=256,
                    in_channels=128,
                    # feat_channels=16,
                    feat_channels=32,
                    out_channels=128,
                    input_feat_shape=7,
                    with_proj=True,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ],
        mask_head=[
            dict(
                type='DynamicMaskHead',
                num_classes=CLASSES,
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    # in_channels=256,
                    # feat_channels=64,
                    # out_channels=256,
                    in_channels=128,
                    # feat_channels=16,
                    feat_channels=32,
                    out_channels=128,
                    input_feat_shape=14,
                    with_proj=False,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                dropout=0.0,
                num_convs=4,
                roi_feat_size=14,
                # in_channels=256,
                in_channels=128,
                conv_kernel_size=3,
                # conv_out_channels=256,
                conv_out_channels=128,
                class_agnostic=False,
                norm_cfg=dict(type='BN'),
                upsample_cfg=dict(type='deconv', scale_factor=2),  # bilinear
                loss_dice=dict(type='DiceLoss', loss_weight=8.0)) for _ in range(num_stages)]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),  # weight = 5.0
                sampler=dict(type='PseudoSampler'),
                pos_weight=1,
                mask_size=28,
                debug=False) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None,
                  rcnn=dict(max_per_img=num_proposals, mask_thr_binary=0.5, nms=dict(type='nms', iou_threshold=0.7))))

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)    # （1）lr=0.00035；（2）lr=0.00025  0.000015
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
# lr_config = dict(policy='step', step=[8, 11], warmup_iters=1000)
lr_config = dict(policy='step', step=[32, 44], warmup_iters=1500)
# lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, min_lr_ratio=1e-4)
total_epochs = 48

# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]
