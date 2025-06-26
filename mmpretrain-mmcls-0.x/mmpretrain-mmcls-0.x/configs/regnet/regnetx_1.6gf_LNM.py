model = dict(
    type='ImageClassifier',
    backbone=dict(type='RegNet', arch='regnetx_1.6gf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=2,
        in_channels=912,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='ReLU'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, )))
dataset_type = 'InnoGastricCancer'
data_root = '/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/V26/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
demo_policies = [
    dict(type='Posterize', bits=4, prob=1.0),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(-30, 30)),
    dict(type='Contrast', prob=0.2, magnitude=0.15),
    dict(type='Sharpness', prob=0.2, magnitude=0.15),
    dict(type='Equalize', prob=0.2)
]
size = 512
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=512),
    dict(type='Resize', size=512),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(
        type='RandAugment',
        policies=[
            dict(type='Posterize', bits=4, prob=1.0),
            dict(
                type='Rotate',
                magnitude_key='angle',
                magnitude_range=(-30, 30)),
            dict(type='Contrast', prob=0.2, magnitude=0.15),
            dict(type='Sharpness', prob=0.2, magnitude=0.15),
            dict(type='Equalize', prob=0.2)
        ],
        num_policies=4,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(512, -1)),
    dict(type='CenterCrop', crop_size=512),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='InnoGastricCancer',
        data_prefix=
        '/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/V26/train',
        ann_file=
        '/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/V26/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=512),
            dict(type='Resize', size=512),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
            dict(
                type='RandAugment',
                policies=[
                    dict(type='Posterize', bits=4, prob=1.0),
                    dict(
                        type='Rotate',
                        magnitude_key='angle',
                        magnitude_range=(-30, 30)),
                    dict(type='Contrast', prob=0.2, magnitude=0.15),
                    dict(type='Sharpness', prob=0.2, magnitude=0.15),
                    dict(type='Equalize', prob=0.2)
                ],
                num_policies=4,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='InnoGastricCancer',
        data_prefix=
        '/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/V26/val',
        ann_file=
        '/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/V26/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(512, -1)),
            dict(type='CenterCrop', crop_size=512),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='InnoGastricCancer',
        data_prefix=
        #'/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/V26/test',
        '/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/outsideV1/val',
        ann_file=
        #'/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/V26/test.txt',
        '/media/inno-bj1/SATA/processed_pad_crop_data/鼓楼胃癌项目_Nx数据20250113_all_review_框不扩大/outsideV1/val_医生筛后删除部分错误样本.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(512, -1)),
            dict(type='CenterCrop', crop_size=512),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(
    interval=1,
    metric=['accuracy', 'precision', 'recall', 'f1_score', 'specificity'],
    metric_options=dict(topk=(1, )))
convert_best_basis = 'recall'
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1,
    min_lr_ratio=1e-05)
optimizer = dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = '/home/inno-bj1/projects/mmcls-inno/pretrained_weights/regnetx-1.6gf_8xb128_in1k_20211213-d1b89758.pth'
workflow = [('train', 1)]
work_dir = '/media/inno-bj1/SATA/work-dir/mmcls/LNM/v26/20250121-2_reg1.6gf_lnm_512*512_ce1_bs16_adamw_lr5e-4_weightdecay1e-4'
gpu_ids = range(0, 1)
