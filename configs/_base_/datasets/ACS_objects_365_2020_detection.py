# dataset settings
dataset_type = 'CocoDataset'
data_root = '/content/gdrive/MyDrive/Research/datasets/objects_365_2020/'
classes = (
    'Watch', 'Book', 'Cosmetics Mirror',  # Human & Accessory
    'Chair', 'Couch', 'Dinning Table', 'Bed', 'Lamp', 'Flower', 'Mirror', 'Side Table', 'Desk',  # Living room
    'Potted Plant', 'Clock', 'Radiator', 'Storage box', 'Vase', 'Air Conditioner', 'Hanger',  # Living room
    'Nightstand', 'Cabinet/shelf', 'Pillow', 'Power outlet', 'Coffee Table', 'Picture/Frame',  # Living room
    'Stool', 'Carpet', 'Fan',  # Living room - Bench, Candle, Napkin
    'Kettle', 'Induction Cooker', 'Oven', 'Extractor', 'Rice Cooker', 'Pot', 'Gas stove', 'Flask',  # Kitchen
    'Toaster', 'Coffee Machine', 'Microwave', 'Blender', 'Refrigerator', 'Dishwasher',  # Kitchen
    'Telephone',  # Office supplies & Tools
    'Camera', 'Extention Cord', 'Speaker', 'Keyboard', 'earphone', 'Moniter/TV', 'Head Phone',  # Electronics
    'Tablet', 'Cell Phone', 'Surveillance Camera', 'Projector', 'Router/modem', 'Computer Box',  # Electronics
    'Printer', 'Laptop', 'Remote',  # Electronics
    'Faucet', 'Soap', 'Sink', 'Toilet', 'Washing Machine/Drying Machine', 'Towel', 'Bathtub', 'Mop',  # Bathroom
    'Toiletry', 'Urinal', 'Showerhead', 'Hair Dryer', 'Broom'  # Bathroom
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
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_validation.json',
        img_prefix=data_root + 'images/validation/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_validation.json',
        img_prefix=data_root + 'images/validation/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
