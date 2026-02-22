import numpy as np
import pytest

from pynegative.io.lens_db_xml import LensDatabase
from pynegative.processing.lens import apply_lens_correction, vignette_kernel

MOCK_VIGNETTE_DB = """
<lensdatabase>
    <lens>
        <maker>TestMaker</maker>
        <model>TestLens</model>
        <calibration>
            <vignetting model="pa" focal="50" aperture="1.8" distance="10" k1="0.1" k2="0.0" k3="0.0" />
            <vignetting model="pa" focal="50" aperture="4.0" distance="10" k1="0.01" k2="0.0" k3="0.0" />
        </calibration>
    </lens>
</lensdatabase>
"""


@pytest.fixture
def vignette_db(tmp_path):
    db_dir = tmp_path / "lensfun"
    db_dir.mkdir()
    (db_dir / "test.xml").write_text(MOCK_VIGNETTE_DB)
    db = LensDatabase()
    db.load(db_dir)
    return db


def test_vignette_params_extraction(vignette_db):
    lens = vignette_db.lenses[0]

    # Test exact match
    params = vignette_db.get_vignette_params(
        lens, focal_length=50, aperture=1.8, distance=10
    )
    assert params is not None
    assert params["k1"] == 0.1

    # Test another exact match
    params = vignette_db.get_vignette_params(
        lens, focal_length=50, aperture=4.0, distance=10
    )
    assert params is not None
    assert params["k1"] == 0.01


def test_vignette_kernel():
    # Create a 100x100 white image
    img = np.ones((100, 100, 3), dtype=np.float32)

    # Apply vignette (brighten corners)
    # k1 = 1.0 means at corners (rn2=1.0), gain is 2.0
    vignette_kernel(img, k1=1.0, k2=0.0, k3=0.0, cx=50, cy=50, full_w=100, full_h=100)

    # Center should be unchanged
    assert img[50, 50, 0] == pytest.approx(1.0)

    # Corners should be brighter
    # rn2 at corner (0,0) relative to (50,50) is (50^2 + 50^2) / (50^2 + 50^2) = 1.0
    assert img[0, 0, 0] == pytest.approx(2.0)
    assert img[99, 99, 0] == pytest.approx(
        1.9604, abs=0.05
    )  # 99 is not exactly 100 but close


def test_apply_lens_correction_vignette():
    img = np.ones((100, 100, 3), dtype=np.float32)
    settings = {"lens_vignette": 0.5, "lens_enabled": True}

    # Apply manual vignette
    res = apply_lens_correction(img, settings, full_size=(100, 100))

    # Corner (0,0) with k1=0.5 should be 1.5
    assert res[0, 0, 0] == pytest.approx(1.5)
