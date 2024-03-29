from pathlib import Path
import tempfile
import joblib

import requests
import pytest
from wsi import WholeSlideImage
import numpy as np


mem = joblib.Memory("cache", verbose=0)


@pytest.fixture(scope="session")
@mem.cache
def get_test_slide():
    slide_file = Path("GTEX-O5YU-1426.svs")
    if not slide_file.exists():
        url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_file.stem}"
        slide_file = Path(tempfile.NamedTemporaryFile(suffix=".svs").name)

        with open(slide_file, "wb") as file:
            for chunk in requests.get(url, stream=True).iter_content(chunk_size=1024 * 4):
                file.write(chunk)
    else:
        for f in sorted(Path().glob(slide_file.stem + "*")):
            if f != slide_file:
                f.unlink()
    return slide_file


@pytest.mark.wsi
@pytest.mark.slow
def test_whole_slide_image_inference(get_test_slide):
    slide = WholeSlideImage(get_test_slide)
    slide.segment()
    assert len(slide.contours_tissue) == len(slide.holes_tissue)
    slide.tile()
    feats, coords = slide.inference("resnet18")

    # Assert conditions
    assert coords.shape == (646, 2), "Coords shape mismatch"
    assert np.allclose(feats.sum(), 14.375092, atol=1e-3), "Features sum mismatch"
