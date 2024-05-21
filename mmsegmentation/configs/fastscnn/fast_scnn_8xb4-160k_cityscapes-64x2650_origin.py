_base_ = [
    '../_base_/models/fast_scnn_origin.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (64, 1024)
data_preprocessor = dict(size=crop_size)


model = dict(data_preprocessor=data_preprocessor)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(64, 2650),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Resize', scale=(64, 2650), keep_ratio=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',  
         scale=(64, 2650),
         keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


# Re-config the data sampler.

train_dataloader = dict(batch_size=4, 
                        num_workers=4, 
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=dict(
                            data_prefix=dict(img_path='original/train_rie', seg_map_path='filter/train'),
                            pipeline=train_pipeline))
                   



val_dataloader = dict(batch_size=4, 
                        num_workers=4, 
                        dataset=dict(
                            data_prefix=dict(img_path='original/train_rie', seg_map_path='filter/train'),
                            pipeline = test_pipeline))
# train_dataloader = dict(batch_size=4, num_workers=4)
# val_dataloader = dict(batch_size=4, num_workers=4)
                        
test_dataloader = val_dataloader

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
