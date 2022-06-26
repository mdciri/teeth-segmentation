import torch
import monai.transforms as T


def get_transorms(
    new_shape, 
    bright_range=None, 
    rotation_range=None, 
    scale_range=None, 
    num_classes=None, 
    to_tensor=True, 
    probs=[0.5, 0.5, 0.5]):

    bright_prob, rot_prob, scale_prob = probs
    transform_list = []

    transform_list.append(T.NormalizeIntensityd(
        keys="img", 
        nonzero=True, 
        channel_wise=True)
    )

    transform_list.append(T.Resized(
        keys=["img", "seg"], 
        spatial_size=new_shape, 
        mode=["bilinear", "nearest"])
    )
    
    if bright_range is not None:
        transform_list.append(T.RandAdjustContrastd(
            keys="img", 
            prob=bright_prob, 
            gamma=bright_range)
        )

    if rotation_range is not None:
        transform_list.append(T.RandRotated(
            keys=["img", "seg"], 
            prob=rot_prob, 
            range_x=rotation_range, 
            mode=["bilinear", "nearest"])
        )
    if scale_range is not None:
        transform_list.append(T.RandZoomd(
            keys=["img", "seg"], 
            prob=scale_prob, 
            min_zoom=scale_range[0], 
            max_zoom=scale_range[1], 
            mode=["bilinear", "nearest"])
        )
    if num_classes is not None:
        transform_list.append(T.AsDiscreted(keys="seg", to_onehot=num_classes))

    if to_tensor == True:
        transform_list.append(T.ToTensord(keys=["img", "seg"], dtype=torch.float32))
        transform_list.append(T.EnsureTyped(keys=["img", "seg"], dtype=torch.float32))

    return T.compose.Compose(transform_list)