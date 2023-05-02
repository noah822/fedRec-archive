



#### Mutliviewed Dataset Wrapper

We provide a wrapper function which converts a normal dataset returning one view per instance in a batch to a multiviewed dataset. Views of instance are generated using the tranform parameters specified.

```python
# pipeline
from utils.augmentation import multiview_dataset_wrapper
from torchvision.transforms import transforms
from torch.utils.data import Dataset
transform = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  ...
])
@multiview_dataset_wrapper(transform, n_view=2, keep_original=True)
class MyDataset(Dataset):
  # your implementation goes here
  ...
 
# use .unpack() & .pack() method to toggle between single and multiviewed dataset
dataset = MyDataset() # multiviewed 
dataset.unpack() # single-viewed
```

