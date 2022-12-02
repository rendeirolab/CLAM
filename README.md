CLAM
====
This is a fork of the repository from [Mahmood lab's CLAM repository](https://github.com/mahmoodlab/CLAM).
It is made available under the GPLv3 License and is available for non-commercial academic purposes.


## Changes from original repository

The purpose of the fork is to compartimentalize the features related with processing of whole-slide images (WSI) from the CLAM model.

The package has been renamed to `wsi_core` as that was the name of the module related with whole slide image processing.

## Usage

```python
from pathlib import Path

import requests
import h5py
import numpy as np
import pandas as pd

from wsi_core import WholeSlideImage

# Get example slide image
slide_name = "GTEX-1117F-1126"
slide_file = Path(f"{slide_name}.svs")
if not slide_file.exists():
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_name}"
    with open(slide_file, "wb") as handle:
        req = requests.get(url)
        handle.write(req.content)

# Instantiate slide class
slide = WholeSlideImage(slide_file)

# Segment tissue
url = "https://raw.githubusercontent.com/mahmoodlab/CLAM/master/presets/bwh_biopsy.csv"
params = pd.read_csv(url).squeeze()
slide.segmentTissue(seg_level=2, filter_params=params.to_dict())
slide.saveSegmentation()

# Visualize segmentation
slide.initSegmentation()
slide.visWSI(vis_level=2).save(f"{slide_name}.segmentation.png")

# Generate coordinates for tiling in h5 file (highest resolution, non-overlapping tiles)
slide.createPatches_bag_hdf5(slide_file.parent, patch_size=224, step_size=224)
slide.get_tile_coordinates()
slide.get_tile_images()
```
from wsi_core.dataset_h5 import Whole_Slide_Bag



## Reference
Please cite the [paper of the original authors](https://www.nature.com/articles/s41551-020-00682-w):

Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

```bibtex
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```
