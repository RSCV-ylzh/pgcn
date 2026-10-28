# The new configuration inherits from the base configuration with the necessary changes
# Source: configs/my_models/sclgt1_faster-rcnn_r50_fpn_2x_hrrsd.py


# import ylzh modules
custom_imports = dict(
    imports=['projects.DSS.modules.o2_loss', ],
    allow_failed_imports=False)
# ylzh

_base_ = [
    '../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]

# super parameters
IMmean=[92.499, 93.356, 86.067]
IMstd=[55.008, 52.668, 51.945]

model = dict(
    data_preprocessor=dict(mean=IMmean, std=IMstd), # for dataset hrrsd
    roi_head=dict(
        bbox_head=dict(
            num_classes=13, # dataset hrrsd
            loss_bbox=dict(type='L1LossOrder2XW', loss_weight=1.0),
        ),
    ),
)

# change dataset hrrsd related settings
dataset_type='CocoDataset'
data_root1='../datasets/hrrsd/'
metainfo1 = {
    'classes': ('bridge','airplane','ground track field',
    'vehicle','parking lot','T junction','baseball diamond',
    'tennis court','basketball court','ship','crossroad','harbor',
    'storage tank'),
}

# change gpu efficiency
BchSz=4
NmbWk=4

train_dataloader = dict(
    batch_size=BchSz,
    num_workers=NmbWk,
    dataset=dict(
        type=dataset_type,
        data_root=data_root1,
        metainfo=metainfo1,
        ann_file='_coco/Anns/hrrsd_train_m-fld_4352_3084.json',
        data_prefix=dict(img='_imgs/'),
    ),
)
val_dataloader = dict(
    batch_size=BchSz,
    num_workers=NmbWk,
    dataset=dict(
        type=dataset_type,
        data_root=data_root1,
        metainfo=metainfo1,
        ann_file='_coco/Anns/hrrsd_val_m-fld_4352_3084.json',
        data_prefix=dict(img='_imgs/'),
    ),
)
test_dataloader = dict(
    batch_size=BchSz,
    num_workers=NmbWk,
    dataset=dict(
        type=dataset_type,
        data_root=data_root1,
        metainfo=metainfo1,
        ann_file='_coco/Anns/hrrsd_test_m-fld_13057_3084.json',
        data_prefix=dict(img='_imgs/'),
    ),
)

# change evaluation related configures
val_evaluator = dict(ann_file=data_root1 + '_coco/Anns/hrrsd_val_m-fld_4352_3084.json')
test_evaluator = dict(ann_file=data_root1 + '_coco/Anns/hrrsd_test_m-fld_13057_3084.json')

# --auto-scale-lr: new fashion on changing learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=True, base_batch_size=16)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1),
]
# --auto-resume: resume from the latest checkpoint in the work_dir 
#                automatically.
