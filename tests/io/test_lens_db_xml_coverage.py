import xml.etree.ElementTree as ET
from pynegative.io.lens_db_xml import (
    LensDatabase,
    load_database,
    get_instance,
    is_database_available,
)


def test_lens_db_coverage(tmp_path):
    # Create fake XML
    db_path = tmp_path / "lensdb"
    db_path.mkdir()
    xml_file = db_path / "test.xml"

    root = ET.Element("lensdatabase")
    cam = ET.SubElement(root, "camera")
    ET.SubElement(cam, "maker").text = "Canon"
    ET.SubElement(cam, "model").text = "Canon EOS 5D"

    lens = ET.SubElement(root, "lens")
    ET.SubElement(lens, "maker").text = "Canon"
    ET.SubElement(lens, "model").text = "Canon EF 50mm"
    ET.SubElement(lens, "crop_factor").text = "1.0"
    ET.SubElement(lens, "focal_min").text = "50"
    ET.SubElement(lens, "focal_max").text = "50"
    ET.SubElement(lens, "aperture").text = "1.8"

    calib = ET.SubElement(lens, "calibration")
    ET.SubElement(
        calib, "distortion", model="ptlens", focal="50", a="0.01", b="-0.02", c="0.03"
    )
    ET.SubElement(
        calib,
        "tca",
        model="poly3",
        focal="50",
        vr0="1",
        vr1="0",
        vr2="0",
        vb0="1",
        vb1="0",
        vb2="0",
    )
    ET.SubElement(
        calib,
        "vignetting",
        model="pa",
        focal="50",
        aperture="1.8",
        distance="10",
        k1="-0.5",
    )

    tree = ET.ElementTree(root)
    tree.write(xml_file)

    db = LensDatabase()
    assert db.load(db_path)

    lens_dict = db.lenses[0]

    assert db.get_distortion_params(lens_dict, 50) is not None
    assert db.get_tca_params(lens_dict, 50) is not None
    assert db.get_vignette_params(lens_dict, 50, 1.8) is not None

    assert db.find_lens("Canon", "Canon EOS 5D", "Canon EF 50mm", 50, 1.8) is not None
    assert db.search_lenses("Canon EF")
    assert db.get_all_lens_names()

    load_database(db_path)
    assert get_instance() is not None
    assert is_database_available()
