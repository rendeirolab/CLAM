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
    slide_file = "GTEX-O5YU-1426"
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_file}"
    path = Path(tempfile.NamedTemporaryFile().name)

    with open(path, "wb") as file:
        for chunk in requests.get(url, stream=True).iter_content(chunk_size=1024):
            file.write(chunk)
    return path


@pytest.mark.wsi
@pytest.mark.slow
def test_whole_slide_image_inference(get_test_slide):
    slide = WholeSlideImage(get_test_slide)
    slide.segment()
    slide.tile()
    feats, coords = slide.inference("resnet18")

    # Assert conditions
    assert coords.shape == (658, 2), "Coords shape mismatch"
    assert np.allclose(feats.sum(), 14.64019), "Features sum mismatch"
