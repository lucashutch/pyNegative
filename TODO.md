# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- **User Presets**: Allow saving and applying named adjustment presets.
- **Metadata Panel**: Display EXIF data (ISO, Shutter, Aperture) in the editor. there should be a button to toggle the panel on and off. it should be located in the top bar. when the panel is closed, the image should take up the full width of the editor.
- **update documentation** - the documentation is out of date with the current state of the project.

## Performance Optimisation

- **Persistent Thumbnail Cache**: Store thumbnails on disk to speed up gallery loading.
- **General Code Cleanup**: Analyse the codebase for redundant, duplicate or unused code.
- **compile numba kernels on startup** - this should speed up the initialisation of the application. the kernels are cached but only compile jit whent hey are needed. should recompute them on startup if they are needed.

## Bugs

- **ad maximum CI test time of 2 mins** - the CI tests are occasionanly hanging. add a timeout, but also investigate why occasioanlly the CI (especailly for py3.14 hangs or takes a super long time at the installing dependancies uv sync step)
-
## Testing Improvement Areas

- **all alreas nee more tests!**
- **generate coverage report** - generate a coverage report to see where we need to add more tests. and add coverage badge to the radme if possible
