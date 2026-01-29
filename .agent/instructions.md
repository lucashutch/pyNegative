# Project Instructions for AI Agents

These are the core engineering standards and operational rules for the `pyNegative` project. Please follow these strictly.

## Git Workflow
- **Never push to `main`**: All changes must be made on a feature branch.
- **Pull Requests**: Once a task is complete, push the branch and create a Pull Request for review.
- **Commit Messages**: Use concise, descriptive messages (following conventional commits if possible, e.g., `feat:`, `fix:`, `refactor:`).

## Environment Management
- **Always use a virtual environment (`venv`)**: Ensure you are working within the local `venv/` directory.
- **Dependency Management**: Add new dependencies to `pyproject.toml` instead of a `requirements.txt`.
- **Installation**: Use `pip install -e .` for development.

## Project Structure
- This project follows the `src-layout`.
- Core logic: `src/pynegative/core.py`
- CLI: `src/pynegative/cli.py`
- UI: `src/pynegative/ui.py`
- Tests: `tests/`
