CLAM
====
This is a fork of the repository from [Mahmood lab's CLAM repository](https://github.com/mahmoodlab/CLAM).
It is made available under the GPLv3 License and is available for non-commercial academic purposes.


## Changes from original repository

The purpose of the fork is to compartimentalize the features related with processing of whole-slide images (WSI) from the CLAM model.

The package has been renamed to `wsi_core` as that was the name of the module related with whole slide image processing.

## Usage

```python
from wsi_core import WholeSlideImage

slide = WholeSlideImage('slide.svs')

# Segment tissue
url = "https://raw.githubusercontent.com/mahmoodlab/CLAM/master/presets/bwh_biopsy.csv"
params = pd.read_csv(url).squeeze()
slide.segmentTissue(seg_level=2, filter_params=params.to_dict())
slide.saveSegmentation('slide.segmentation.pickle')

# Visualize segmentation
slide.initSegmentation('slide.segmentation.pickle')  # load segmentation
slide.visWSI(vis_level=2).save("slide.segmentation.png")

# Generate coordinates for tiling
slide.process_contours('.', patch_size=512, step_size=512)

# Read one tile
import h5py
h5 = h5py.File('./slide.h5')
params = dict(h5["coords"].attrs)
kwargs = dict(
    level=params["patch_level"], size=(params["patch_size"], params["patch_size"])
)
coords = h5["coords"][()]

slide.wsi  # openslide instance of the slide
tile = slide.wsi.read_region(coords[0], **kwargs)[..., :-1] / 255
```

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
