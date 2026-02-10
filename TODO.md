# Roadmap & Future Improvements

This document tracks planned features, project goals, and areas for technical improvement.

## Roadmap / TODO

- **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- **Kelvin White Balance**: Map relative temperature to absolute Kelvin values based on RAW metadata.
- **User Presets**: Allow saving and applying named adjustment presets.

## Performance Optimisation

- **General Code Cleanup**: Analyse the codebase for redundant, duplicate or unused code.
- **~~Compile Numba Kernels on Startup~~**: ✅ Done — kernels are pre-compiled behind the splash screen on first launch.

## Bugs

- **Add Maximum CI Test Time of 2 mins**: The CI tests occasionally hang. Add a timeout, and investigate why the CI (especially for py3.14) sometimes hangs or takes a very long time at the `uv sync` step.

## Testing Improvement Areas

- **All areas need more tests!**
- **Generate Coverage Report**: Generate a coverage report to see where we need to add more tests, and add a coverage badge to the README if possible.
