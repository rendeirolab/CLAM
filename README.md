WSI
====

[![Documentation Status](https://readthedocs.org/projects/wsi/badge/?version=latest&style=flat-square)](https://wsi.readthedocs.io/en/latest)


This is a fork of the repository from [Mahmood lab's CLAM repository](https://github.com/mahmoodlab/CLAM).
It is made available under the GPLv3 License and is available for non-commercial academic purposes.


## Changes from original repository

The purpose of the fork is to compartimentalize the features related with processing of whole-slide images (WSI) from the CLAM model.

The package has been renamed to `wsi`.


## Installation

While the repository is private, make sure you [exchange SSH keys of the machine with Github.com](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Then simply install with `pip` (make sure you have a recent enough version with `pip install pip -U`:
```bash
pip install git+https://github.com/rendeirolab/wsi.git
```

Note that the package uses setuptols-scm for version control and therefore the installation source needs to be a git repository (a zip file of source code won't work).

## Usage

The only exposed class is `WholeSlideImage` enables all the functionalities of the package.

### Quick start - segmentation, tiling and feature extraction
```python
from wsi import WholeSlideImage    

url = "https://brd.nci.nih.gov/brd/imagedownload/GTEX-O5YU-1426"
slide = WholeSlideImage(url)
slide.segment()
slide.tile()
feats, coords = slide.inference("resnet18")
```

### Full example

This package is meant for both interactive use and for use in a pipeline at scale.
By default actions do not return anything, but instead save the results to disk in files relative to the slide file.

All major functions have sensible defaults but allow for customization.
Please check the docstring of each function for more information.

```python
from wsi import WholeSlideImage
from wsi.utils import Path

# Get example slide image
slide_file = Path("GTEX-12ZZW-2726.svs")
if not slide_file.exists():
    import requests
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_file.stem}"
    with open(slide_file, "wb") as handle:
        req = requests.get(url)
        handle.write(req.content)

# Instantiate slide object
# # from a local file
slide = WholeSlideImage(slide_file)
# # from a URL (will be saved in temporary folder)
slide = WholeSlideImage("https://brd.nci.nih.gov/brd/imagedownload/GTEX-O5YU-1426")
# # instantiation can be done with custom attributes as well
slide = WholeSlideImage(slide_file, attributes=dict(donor="GTEX-12ZZW", tissue='Ileum', sex='Male'))

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
loader = slide.as_data_loader(with_coords=True)

# Extract features "manually"
import torch
from tqdm import tqdm
model = torch.hub.load("pytorch/vision", "resnet18", weights="DEFAULT")
feats = list()
coords = list()
for count, (batch, yx) in tqdm(enumerate(loader), total=len(loader)):
    with torch.no_grad(): 
        f = model(batch).numpy()
    feats.append(f)
    coords.append(yx)

feats = np.concatenate(feats, axis=0)
coords = np.concatenate(coords, axis=0)

# Extract features "automatically"
feats, coords = slide.inference('resnet18')

# Additional parameters can also be specified
feats, coords = slide.inference('resnet18', device='cuda', data_loader_kws=dict(batch_size=512))

# Generate a torch_geometric data object
gdata = slide.as_torch_geometric_data(feats, coords)  # from existing features and coordinates
gdata = slide.as_torch_geometric_data(model_name='resnet18')  # without
```

## Development

`wsi` will not be providing new features and future maintenance will be restricted to bugs. Please refer to our newer project [Lazyslide](https://github.com/rendeirolab/LazySlide) for a more modern, interoperable package for the analysis of whole-slide images.


To contribute bug maintenance, clone the repository, open an issue in the [issue tracker](https://github.com/rendeirolab/wsi/issues), and submit a pull request.
```
git clone git@github.com:rendeirolab/wsi.git
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
<!-- For the current code, please cite also the following paper: -->
