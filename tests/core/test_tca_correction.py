import pytest
import numpy as np
from pynegative.io.lens_db_xml import LensDatabase
from pynegative.processing.lens import remap_tca_distortion_kernel

MOCK_TCA_DB = """
<lensdatabase>
    <lens>
        <maker>TestMaker</maker>
        <model>TestLens</model>
        <mount>TestMount</mount>
        <calibration>
            <tca model="poly3" focal="18" vr0="1.01" vr1="0.0" vr2="0.0" vb0="0.99" vb1="0.0" vb2="0.0" />
            <tca model="poly3" focal="50" vr0="1.02" vr1="0.0" vr2="0.0" vb0="0.98" vb1="0.0" vb2="0.0" />
        </calibration>
    </lens>
</lensdatabase>
"""


@pytest.fixture
def tca_db(tmp_path):
    db_dir = tmp_path / "lensfun"
    db_dir.mkdir()
    (db_dir / "tca.xml").write_text(MOCK_TCA_DB)
    db = LensDatabase()
    db.load(db_dir)
    return db


def test_tca_param_extraction(tca_db):
    lens = tca_db.lenses[0]

    # Test exact focal length
    params = tca_db.get_tca_params(lens, 18.0)
    assert params["vr0"] == 1.01
    assert params["vb0"] == 0.99

    # Test interpolation
    params = tca_db.get_tca_params(lens, 34.0)
    # (34-18)/(50-18) = 16/32 = 0.5
    # vr0 = 1.01 + 0.5 * (1.02 - 1.01) = 1.015
    assert pytest.approx(params["vr0"]) == 1.015
    assert pytest.approx(params["vb0"]) == 0.985


def test_tca_kernel_execution():
    # Create a small test image with a white square in the middle
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[40:60, 40:60, :] = 1.0

    out = np.zeros_like(img)

    # Params: No distortion, just extreme TCA to see the shift
    model_type = 1  # poly3
    dist_p = np.array([0.0], dtype=np.float32)  # k1=0
    tca_red = np.array([1.1, 0.0, 0.0], dtype=np.float32)  # Scale Red by 1.1
    tca_blue = np.array([0.9, 0.0, 0.0], dtype=np.float32)  # Scale Blue by 0.9

    cx, cy = 50.0, 50.0
    fw, fh = 100.0, 100.0

    remap_tca_distortion_kernel(
        img, out, model_type, dist_p, tca_red, tca_blue, cx, cy, fw, fh, 1.0
    )

    # Check that channels are shifted
    # Red should be scaled by 1.1 (pushed OUT from center, square CONTRACTS)
    # Green (Reference) stays at 40:60
    # Blue scaled by 0.9 (sampled from closer in, square EXPANDS)

    # At x=60 (dx=10):
    # Green: samples from 60 (edge), should be ~0.5 or 1.0
    # Red: samples from 50 + 10 * 1.1 = 61 (black) -> 0.0
    # Blue: samples from 50 + 10 * 0.9 = 59 (white) -> 1.0

    assert out[50, 61, 0] < 0.1  # Red (was black, still black)
    assert (
        out[50, 59, 0] < 0.1
    )  # Red (was white, now samples from 50+9*1.1 = 59.9 = black edge)

    assert out[50, 59, 1] > 0.9  # Green (stays white)
    assert out[50, 61, 1] < 0.1  # Green (stays black)

    assert (
        out[50, 60, 2] > 0.9
    )  # Blue (was black, now samples from 50+10*0.9 = 59 = white)

    assert np.any(out[:, :, 0] != out[:, :, 1])  # TCA happened
