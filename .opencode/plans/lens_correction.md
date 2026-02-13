# Lens Detection & Correction — Revised Implementation Plan

An EXIF-first approach using directly-downloaded lensfun database XML files, avoiding GPL/LGPL licensing issues while providing robust lens correction capabilities.

## Architecture Overview

### Detection Priority (3-Tier)

1. **Embedded RAW Metadata** — Newer cameras embed lens correction factors directly in RAW/EXIF data
   - Parse lens model name and correction coefficients from EXIF
   - Use embedded profiles when available (highest priority, most accurate)

2. **Lensfun Database Match** — Fall back to downloaded lensfun XML database
   - Database downloaded during installation to `~/.local/share/pyNegative/data/lensfun/`
   - Parse XML directly (no external library dependency)
   - Auto-match based on camera make/model + lens model from EXIF

3. **Manual Selection** — User-selectable lens from searchable dropdown
   - Display when auto-detection fails or user wants to override
   - Searchable by lens name, focal length, manufacturer
   - Works even without database (manual sliders always available)

### Graceful Degradation

The app **never crashes** if the lens database is missing:
- Phase 1: Try embedded EXIF data
- Phase 2: Try lensfun database (if present)
- Phase 3: Show manual controls with "Database not found" message
- Manual sliders for distortion/vignetting/CA always work

---

## Phase 1: Lens Database Infrastructure

### [NEW] `scripts/download_lens_database.py`
- Download lensfun database XML files from GitHub
- URL: `https://raw.githubusercontent.com/lensfun/lensfun/master/data/db/`
- Downloads all XML files to specified directory
- Creates `.lensdb_version` file with timestamp
- Usage: `python scripts/download_lens_database.py --output-dir ~/.local/share/pyNegative/data/lensfun/`
- **Not included in repo** — devs must run manually or via install script

### [MODIFY] `scripts/install-pynegative.sh`
- After installing app, call `download_lens_database.py`
- Install to: `$INSTALL_DIR/data/lensfun/`
- Skip gracefully if download fails (warn user but don't fail install)
- Store path in config for app to find

### [MODIFY] `scripts/install-pynegative.bat` (Windows)
- Same as above for Windows
- Use PowerShell or Python to download

### [NEW] `src/pynegative/io/lens_db_xml.py`
- Parse lensfun XML files directly using `xml.etree.ElementTree`
- Functions:
  - `load_database(db_path)` — load all XML files from directory
  - `find_lens_by_model(camera_make, camera_model, lens_model)` — fuzzy match
  - `search_lenses(query)` — search for manual selection
  - `get_lens_correction_data(lens_id, focal_length, aperture)` — get k1, k2, etc.
  - `is_database_available()` — check if DB exists
- Cache loaded database in memory
- **Graceful handling**: Return `None` if DB not found, UI shows appropriate message

### [NEW] `src/pynegative/io/lens_metadata.py`
- Extract embedded lens data from RAW/EXIF:
  - `extract_lens_from_exif(raw_path)` — parse lens model from EXIF tags
  - `extract_distortion_coefficients(raw_path)` — get embedded distortion data (Sony, Olympus, etc.)
  - `extract_vignetting_data(raw_path)` — get embedded vignette data
  - `extract_ca_data(raw_path)` — get embedded CA data
- Support for major RAW formats
- Returns `None` for unsupported formats/cameras

### [NEW] `src/pynegative/io/lens_resolver.py`
- 3-tier priority resolver:
  ```python
  def resolve_lens_profile(raw_path, camera_exif, lens_exif):
      # Tier 1: Try embedded RAW data
      embedded = extract_embedded_profile(raw_path)
      if embedded:
          return ProfileSource.EMBEDDED, embedded
      
      # Tier 2: Try lensfun database
      if lens_db_xml.is_database_available():
          profile = lens_db_xml.find_lens_by_model(...)
          if profile:
              return ProfileSource.LENSFUN_DB, profile
      
      # Tier 3: Return None (manual mode)
      return ProfileSource.MANUAL, None
  ```

### [NEW] `src/pynegative/ui/controls/lens_controls.py`
- `LensControls(BaseControlWidget)` with `CollapsibleSection("LENS")`
- Detection status label:
  - "Using embedded EXIF profile" (green)
  - "Using lens database" (blue)
  - "Manual mode - database not found" (orange)
  - "Manual mode - no profile found" (yellow)
- Searchable camera/lens dropdowns (only when DB available)
- "Auto Detect" button to re-run detection
- Manual sliders:
  - Distortion (-1.0 to +1.0)
  - Vignette (0 to 100)
  - CA (0 to 100)
- Emits `settingChanged` for all lens settings

### [MODIFY] `src/pynegative/ui/widgets/metadata_panel.py`
- Show detected lens name and source
- Show "Update lens database" link if DB missing

### [MODIFY] `pyproject.toml`
- **No new dependencies** — uses only standard library for XML parsing

---

## Phase 2: Distortion Correction

### [NEW] `src/pynegative/processing/lens.py`
- `apply_distortion_correction(img, k1, k2, k3, ...)` — uses OpenCV `cv2.undistort()`
- `apply_lensfun_distortion(img, lens_profile, focal_length)` — apply lensfun params
- `apply_embedded_distortion(img, embedded_params)` — apply embedded EXIF params
- Caches distortion maps for performance
- **Works with any source**: embedded, lensfun DB, or manual k1/k2 values

### [MODIFY] `src/pynegative/ui/pipeline/worker.py`
- Add `_process_lens_stage()` before heavy processing
- Apply distortion correction first
- Pass lens params through pipeline

---

## Phase 3: Vignette Correction

### [MODIFY] `src/pynegative/processing/lens.py`
- `apply_vignette_correction(img, correction_params)`
- Support lensfun vignetting model
- Support manual slider (0-100%)

### [MODIFY] `src/pynegative/ui/controls/lens_controls.py`
- Add Vignette slider + enable toggle

---

## Phase 4: Chromatic Aberration Correction

### [MODIFY] `src/pynegative/processing/lens.py`
- `apply_tca_correction(img, red_params, blue_params)` — lateral CA correction
- Support lensfun TCA model
- Support manual CA slider

### [MODIFY] `src/pynegative/ui/controls/lens_controls.py`
- Add CA slider + enable toggle

---

## Phase 5: Manual Defringe

### [NEW] `src/pynegative/utils/numba_defringe.py`
- `defringe_kernel()` — Numba `@njit(parallel=True)` kernel
- Desaturates purple/green fringes on high-contrast edges

### [MODIFY] `src/pynegative/processing/lens.py`
- `apply_defringe(img, amount)` — wrapper calling Numba kernel

### [MODIFY] `src/pynegative/ui/controls/lens_controls.py`
- Add Defringe slider (0–100)

---

## Task List

### Phase 1: Database Infrastructure
- [x] Create `scripts/download_lens_database.py` to download lensfun XML files
- [x] Modify `scripts/install-pynegative.sh` to download DB during install
- [x] Modify `scripts/install-pynegative.bat` for Windows
- [x] Create `src/pynegative/io/lens_db_xml.py` to parse XML directly
- [x] Create `src/pynegative/io/lens_metadata.py` for EXIF extraction
- [x] Create `src/pynegative/io/lens_resolver.py` for 3-tier priority logic
- [x] Create `src/pynegative/ui/controls/lens_controls.py` with graceful degradation
- [x] Modify metadata panel to show lens info and DB status
- [x] Add sidecar persistence for lens settings
- [x] Tests for XML parsing and lens matching

### Phase 2: Distortion Correction
- [ ] Implement `src/pynegative/processing/lens.py` distortion correction
- [ ] Add lens stage to pipeline worker
- [ ] Integrate with lens controls

### Phase 3: Vignette Correction
- [ ] Implement vignette correction in processing/lens.py
- [ ] Add vignette controls to UI

### Phase 4: Chromatic Aberration
- [ ] Implement TCA correction in processing/lens.py
- [ ] Add CA controls to UI

### Phase 5: Manual Defringe
- [ ] Create Numba defringe kernel
- [ ] Add defringe to processing pipeline
- [ ] Add defringe slider to UI

### Documentation
- [ ] Update README with lens correction features
- [ ] Add LICENSE note about CC-BY-SA lens database
- [ ] Document manual lens correction for unsupported lenses

---

## License Compliance

### Code (MIT)
- All pyNegative code remains MIT licensed
- XML parsing code is original (not derived from lensfun)

### Database (CC-BY-SA 3.0)
- Lens database files downloaded from lensfun GitHub
- Stored separately in `data/lensfun/` directory
- Attribution included in README and LICENSE
- Any modifications to DB must also be CC-BY-SA 3.0

### No GPL/LGPL Dependencies
- No lensfunpy library (GPL/LGPL)
- No lensfun C library
- Pure Python XML parsing only
