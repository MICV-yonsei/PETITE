
import math
import os
# import matplotlib.pyplot as plt
import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
import numpy as np
import nibabel as nib

from utils.utils import distributed_all_gather
from monai.transforms import SaveImaged
from monai.data import decollate_batch
import torch.utils.data.distributed


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])


    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]), #(1,128,128,128)
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityd(
                keys=["image", "label"], minv=0, maxv=1
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld( 
                keys=["image", "label"],
                label_key="label", 
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #=roi
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    ) 
    val_transform = transforms.Compose(
            [   
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),),
                transforms.ScaleIntensityd(
                    keys=["image", "label"], minv=0, maxv=1
                ),
                # transforms.SaveImaged(keys=["image", "label"], output_dir="./outputs/", output_ext=".nii", resample=False),
    
            ]
        )
    ###original###
    
    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
        
        
    else:
        train_files = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader