import pytest
from pathlib import Path
import os
import time
from PIL import Image
from pynegative import core

@pytest.fixture
def temp_raw_path(tmp_path):
    """Provides a temporary RAW file path."""
    raw_file = tmp_path / "test_image.cr2"
    raw_file.touch()
    return raw_file

def test_thumbnail_cache_save_load(temp_raw_path):
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color='red')
    metadata = {"width": 100, "height": 100, "date": "2024-01-01"}
    size = 400

    # Save to cache
    core.save_cached_thumbnail(temp_raw_path, img, metadata, size)

    # Check directory exists
    cache_dir = core.get_thumbnail_cache_dir(temp_raw_path)
    assert cache_dir.exists()
    assert (cache_dir / SIDECAR_DIR if 'SIDECAR_DIR' in locals() else cache_dir).exists() # Already in SIDECAR_DIR

    # Load from cache
    loaded_img, loaded_metadata = core.load_cached_thumbnail(temp_raw_path, size)

    assert loaded_img is not None
    assert loaded_img.size == (100, 100)
    assert loaded_metadata == metadata

def test_thumbnail_cache_invalidation(temp_raw_path):
    img = Image.new('RGB', (100, 100), color='red')
    metadata = {"width": 100, "height": 100}
    size = 400

    # Save initial version
    core.save_cached_thumbnail(temp_raw_path, img, metadata, size)

    # Update mtime of the source file
    original_mtime = temp_raw_path.stat().st_mtime
    new_mtime = original_mtime + 100
    os.utime(temp_raw_path, (new_mtime, new_mtime))

    # Try to load - should fail because mtime changed
    loaded_img, loaded_metadata = core.load_cached_thumbnail(temp_raw_path, size)
    assert loaded_img is None
    assert loaded_metadata == {}

    # Save new version
    core.save_cached_thumbnail(temp_raw_path, img, metadata, size)
    loaded_img, loaded_metadata = core.load_cached_thumbnail(temp_raw_path, size)
    assert loaded_img is not None

def test_rename_sidecar_with_thumbnails(temp_raw_path, tmp_path):
    img = Image.new('RGB', (100, 100), color='blue')
    metadata = {"width": 100, "height": 100}
    size = 200

    core.save_cached_thumbnail(temp_raw_path, img, metadata, size)

    # Rename the raw file
    new_raw_path = tmp_path / "renamed_image.cr2"
    temp_raw_path.rename(new_raw_path)

    core.rename_sidecar(temp_raw_path, new_raw_path)

    # Check if thumbnail cache was renamed
    old_thumb_dir = core.get_thumbnail_cache_dir(temp_raw_path)
    new_thumb_dir = core.get_thumbnail_cache_dir(new_raw_path)

    # Since they are in the same parent tmp_path, old_thumb_dir might still exist
    # but the specific file inside should be renamed.
    # mtime stays the same often on rename, but let's check
    mtime = int(new_raw_path.stat().st_mtime)
    expected_thumb = new_thumb_dir / f"renamed_image.cr2.{mtime}.{size}.webp"
    assert expected_thumb.exists()

    # Load from new location
    loaded_img, loaded_metadata = core.load_cached_thumbnail(new_raw_path, size)
    assert loaded_img is not None
