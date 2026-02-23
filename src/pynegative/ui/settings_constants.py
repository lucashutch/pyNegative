"""
Constants for settings groups and keys used for selective copy/paste.
"""

SETTINGS_GROUPS = {
    "Tone": [
        ("exposure", "Exposure"),
        ("contrast", "Contrast"),
        ("highlights", "Highlights"),
        ("shadows", "Shadows"),
        ("whites", "Whites"),
        ("blacks", "Blacks"),
    ],
    "Color": [
        ("temperature", "Temperature"),
        ("tint", "Tint"),
        ("saturation", "Saturation"),
    ],
    "Detail": [
        ("sharpen_value", "Sharpening"),
        ("denoise_luma", "Luminance Noise"),
        ("denoise_chroma", "Color Noise"),
        ("de_haze", "Dehaze"),
    ],
    "Lens": [
        ("lens_enabled", "Enable Lens Corrections"),
        ("lens_autocrop", "Auto Crop (Lens)"),
        ("lens_distortion", "Distortion"),
        ("lens_vignette", "Vignette"),
        ("lens_ca", "Chromatic Aberration"),
        ("defringe_purple", "Defringe Purple"),
        ("defringe_green", "Defringe Green"),
        ("defringe_edge", "Defringe Edge"),
        ("defringe_radius", "Defringe Radius"),
        ("lens_camera_override", "Camera Override"),
        ("lens_name_override", "Lens Override"),
    ],
    "Geometry": [
        ("crop", "Crop"),
        ("rotation", "Rotation"),
        ("flip_h", "Flip Horizontal"),
        ("flip_v", "Flip Vertical"),
    ],
}
