# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/oidv6/'
classes= (
    'Alarm clock',
    'Bathroom cabinet',
    'Bathtub',
    'Bed',
    'Bookcase',
    'Ceiling fan',
    'Chest of drawers',
    'Clock',
    'Closet',
    'Computer monitor',
    'Curtain',
    'Dishwasher',
    'Drawer',
    'Fireplace',
    'Gas stove',
    'Couch',
    'Infant bed',
    'Microwave oven',
    'Kitchen & dining room table',
    'Sofa bed',
    'Toilet',
    'Washing machine',
    'Wardrobe',
    'Home appliance'
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train_labels.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val_labels.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test_labels.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')