_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='configs/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5,
              topk=(1, ),
              ),
)

data_root = 'flower_dataset'
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',
        data_prefix='val',
    ))
test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.001))

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)