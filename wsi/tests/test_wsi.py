import pytest
from wsi import WholeSlideImage
import numpy as np


@pytest.mark.wsi
@pytest.mark.slow
def test_whole_slide_image_inference():
    slide_file = "GTEX-O5YU-1426"
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_file}"
    slide = WholeSlideImage(url)
    slide.segment()
    slide.tile()
    feats, coords = slide.inference("resnet18")

    # Assert conditions
    assert coords.shape == (658, 2), "Coords shape mismatch"
    assert np.allclose(feats.sum(), 14.64019), "Features sum mismatch"
