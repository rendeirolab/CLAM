CLAM
====
This is a fork of the repository from [Mahmood lab's CLAM repository](https://github.com/mahmoodlab/CLAM).
It is made available under the GPLv3 License and is available for non-commercial academic purposes.


## Changes from original repository

The purpose of the fork is to compartimentalize the features related with processing of whole-slide images (WSI) from the CLAM model.

The package has been renamed to `wsi_core` as that was the name of the module related with whole slide image processing.


## Installation

While the repository is private, make sure you [exchange SSH keys of the machine with Github.com](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Then simply install with `pip`:
```bash
git clone git@github.com:rendeirolab/CLAM.git
cd CLAM
pip install .
```

Note that the package uses setuptols-scm for version control and therefore the installation source needs to be a git repository (a zip file of source code won't work).

## Usage

This package is meant for both interactive use and for use in a pipeline at scale.
By default actions do not return anything, but instead save the results to disk in files relative to the slide file.

All major functions have sensible defaults but allow for customization.
Please check the docstring of each function for more information.

```python
from wsi_core import WholeSlideImage
from wsi_core.utils import Path

# Get example slide image
slide_file = Path("GTEX-12ZZW-2726.svs")
if not slide_file.exists():
    import requests
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_file.stem}"
    with open(slide_file, "wb") as handle:
        req = requests.get(url)
        handle.write(req.content)

# Instantiate slide object
slide = WholeSlideImage(slide_file)

# Instantiate slide object
slide = WholeSlideImage(slide_file, attributes=dict(donor="GTEX-12ZZW"))

# Segment tissue (segmentation mask is stored as polygons in slide.contours_tissue)
slide.segment()

# Visualize segmentation (PNG file is saved in same directory as slide_file)
slide.plot_segmentation()

# Generate coordinates for tiling in h5 file (highest resolution, non-overlapping tiles)
slide.tile()

# Get coordinates (from h5 file)
slide.get_tile_coordinates()

# Get image of single tile using lower level OpenSlide handle (`wsi` object)
slide.wsi.read_region((1_000, 2_000), level=0, size=(224, 224))

# Get tile images for all tiles (as a generator)
images = slide.get_tile_images()
for img in images:
    ...

# Save tile images to disk as individual jpg files
slide.save_tile_images(output_dir=slide_file.parent / (slide_file.stem + "_tiles"))

# Use in a torch dataloader
loader = slide.as_data_loader()

# Extract features
import torch
from tqdm import tqdm
model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True) 
for count, (batch, coords) in tqdm(enumerate(loader), total=len(loader)):
    with torch.no_grad(): 
        features = model(batch).numpy()
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
