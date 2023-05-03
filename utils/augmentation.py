import torchvision.transforms as transforms
import torch.nn as nn


def get_default_img_transforms(input_shape, n_channel=3, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform_list = [transforms.RandomResizedCrop(size=input_shape),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomApply([color_jitter], p=0.8)] 
    if n_channel == 3:
        transform_list.append(
            transforms.RandomGrayscale(p=0.2)
        )
    
    data_transforms = transforms.Compose(transforms)
    return data_transforms


def multiview_dataset_wrapper(transforms, n_view=2, keep_original=False):
    def _wrapper(dataset_cls):
        class _multiview_dataset_wrapper(dataset_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.transforms = transforms
                self.n_view = n_view
                self.keep_original = keep_original
                
                self.multiviewed = True
                
            def __len__(self):
                return super().__len__()
            
            def __getitem__(self, index):
                if not self.multiviewed:
                    return super().__getitem__(index)
                
                data = super().__getitem__(index)
                if len(data) > 1:
                    data = data[0]
                    
                views = [None for _ in range(self.n_view)]
                if self.keep_original:
                    views[0] = data
                    for i in range(1, self.n_view):
                        views[i] = self.transforms(data)
                else:
                    for i in range(self.n_view):
                        views[i] = self.transforms(data)
                return views
            
            def unpack(self):
                self.multiviewed = False
            def pack(self):
                self.multiviewed = True
                
        return _multiview_dataset_wrapper

    return _wrapper

class Augmentation(nn.Module):
    def __init__(self, transforms, n_view=2):
        super(Augmentation, self).__init__()
        self.transform = nn.Sequential(*transforms)
        self.n_view = n_view
    def forward(self, x):
        return [self.transform(x) for _ in range(self.n_view)]
