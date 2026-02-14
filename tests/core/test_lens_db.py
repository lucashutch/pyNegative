import pytest
from pynegative.io.lens_db_xml import LensDatabase

# Mock XML content
MOCK_LENS_DB = """
<lensdatabase>
    <camera>
        <maker>Canon</maker>
        <model>EOS R50</model>
        <mount>Canon RF</mount>
    </camera>
    <lens>
        <maker>GoPro</maker>
        <model>HERO4 Black Edition</model>
        <mount>GoPro</mount>
        <crop_factor>5.6</crop_factor>
        <focal_min>3</focal_min>
        <focal_max>3</focal_max>
        <aperture>2.8</aperture>
    </lens>
    <lens>
        <maker>Canon</maker>
        <model>RF 50mm F1.8 STM</model>
        <mount>Canon RF</mount>
        <crop_factor>1.0</crop_factor>
        <focal_min>50</focal_min>
        <focal_max>50</focal_max>
        <aperture>1.8</aperture>
    </lens>
</lensdatabase>
"""


@pytest.fixture
def lens_db(tmp_path):
    # Create a temporary database structure
    db_dir = tmp_path / "lensfun"
    db_dir.mkdir()
    (db_dir / "compact-canon.xml").write_text(MOCK_LENS_DB)

    db = LensDatabase()
    db.load(db_dir)
    return db


def test_find_lens_empty_model(lens_db):
    """Test that empty lens model returns None instead of a false match."""
    match = lens_db.find_lens(
        camera_maker="Canon",
        camera_model="EOS R50",
        lens_model="",
        focal_length=None,
        aperture=None,
    )
    assert match is None


def test_find_lens_unknown_model(lens_db):
    """Test that unknown lens model returns None."""
    match = lens_db.find_lens(
        camera_maker="Canon",
        camera_model="EOS R50",
        lens_model="Unknown Lens",
        focal_length=18.0,
        aperture=3.5,
    )
    assert match is None


def test_find_lens_valid_match(lens_db):
    """Test that a valid lens can still be found."""
    match = lens_db.find_lens(
        camera_maker="Canon",
        camera_model="EOS R50",
        lens_model="Canon RF 50mm F1.8 STM",
        focal_length=50.0,
        aperture=1.8,
    )
    assert match is not None
    assert match["maker"] == "Canon"
    assert match["model"] == "RF 50mm F1.8 STM"


def test_find_lens_short_model_ignored(lens_db):
    """Test that very short lens models are ignored."""
    match = lens_db.find_lens(
        camera_maker="Canon",
        camera_model="EOS R50",
        lens_model="A",
        focal_length=None,
        aperture=None,
    )
    assert match is None
