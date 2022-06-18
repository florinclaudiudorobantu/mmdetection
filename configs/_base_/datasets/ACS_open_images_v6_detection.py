# dataset settings
dataset_type = 'CocoDataset'
data_root = '/content/gdrive/MyDrive/Research/datasets/open_images_v6/'
classes = (
    'Washing machine', 'Toaster', 'Oven', 'Blender', 'Gas stove', 'Mechanical fan', 'Heater',  # Home appliance
    'Kettle', 'Hair dryer', 'Refrigerator', 'Wood-burning stove', 'Humidifier', 'Mixer',  # Home appliance
    'Coffeemaker', 'Microwave oven', 'Dishwasher', 'Sewing machine', 'Hand dryer',  # Home appliance
    'Ceiling fan', 'Home appliance',  # Home appliance
    'Sink', 'Bidet', 'Shower', 'Tap', 'Bathtub', 'Toilet',  # Plumbing fixture
    'Pillow',  # Entity
    'Spice rack',  # Kitchenware
    'Slow cooker', 'Food processor', 'Waffle iron', 'Pressure cooker',  # Kitchen appliance
    'Fireplace',  # Entity
    'Countertop',  # Entity
    'Book',  # Entity,
    'Chair', 'Cabinetry', 'Desk', 'Wine rack',  # Furniture
    'Sofa bed', 'Loveseat', 'Couch',  # Couch
    'Wardrobe', 'Nightstand', 'Bookcase',  # Furniture
    'Infant bed', 'Studio couch', 'Bed',  # Bed
    'Filing cabinet',  # Furniture
    'Coffee table', 'Kitchen & dining room table', 'Table',  # Table
    'Chest of drawers', 'Cupboard', 'Drawer', 'Stool', 'Shelf', 'Wall clock', 'Bathroom cabinet',  # Furniture
    'Closet',  # Furniture
    'Dog bed',  # Entity
    'Cat furniture',  # Entity
    'Lantern',  # Entity
    'Alarm clock', 'Digital clock', 'Clock',  # Clock
    'Vase',  # Entity
    'Window blind',  # Entity
    'Curtain',  # Entity
    'Mirror',  # Entity
    'Picture frame',  # Entity
    'Lamp',  # Entity
    'Towel', 'Soap dispenser',  # Bathroom accessory
    'Camera',  # Tool
    'Cassette deck', 'Headphones', 'Laptop', 'Computer keyboard', 'Printer', 'Computer mouse',
    'Computer monitor', 'Power plugs and sockets', 'Light switch', 'Television',  # Tool
    'Mobile phone', 'Corded phone', 'Telephone',   # Telephone
    'Tablet computer', 'Microphone', 'Ipod', 'Remote control',  # Tool
    'Houseplant',  # Plant
    'Watch',  # Fashion accessory
    'Flowerpot'  # Container
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
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
