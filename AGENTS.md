# AGENTS.md

## Project Shape
- Python package source lives in `ultralytics/`; the package version is `ultralytics.__version__` and the console scripts `yolo` and `ultralytics` both enter at `ultralytics.cfg:entrypoint`.
- Main task/mode dispatch is in `ultralytics/cfg/__init__.py`; valid tasks are `detect`, `segment`, `classify`, `pose`, `obb`, `semantic`, and valid modes are `train`, `val`, `predict`, `export`, `track`, `benchmark`.
- Model families are lazily exposed from `ultralytics/__init__.py` as `YOLO`, `YOLOWorld`, `YOLOE`, `NAS`, `SAM`, `FastSAM`, and `RTDETR`.
- Default runtime args are in `ultralytics/cfg/default.yaml`; model and dataset YAMLs live under `ultralytics/cfg/models/` and `ultralytics/cfg/datasets/`.
- Docs source is `docs/en/` per `mkdocs.yml`; generated/reference docs are updated by `docs/build_reference.py` and `docs/build_docs.py` in CI.

## Setup And Commands
- Development install: `pip install -e .` or, matching CI, `uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu`.
- CI-style base test install: `uv pip install -e ".[export-base,solutions]" aiohttp pytest-cov pytest-xdist "git+https://github.com/ultralytics/CLIP.git" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match`.
- Quick environment sanity check: `yolo checks`.
- Run the regular test suite like CI: `YOLO_AUTOINSTALL=false pytest -n auto --dist=loadfile --cov=ultralytics/ --cov-report=xml tests/ --export-env base`.
- Run one test file or test: `YOLO_AUTOINSTALL=false pytest tests/test_python.py` or `YOLO_AUTOINSTALL=false pytest tests/test_python.py::test_model_forward`.
- Slow tests are excluded unless `--slow` is passed; use `YOLO_AUTOINSTALL=false pytest --slow tests/ --export-env base` only when the extra runtime/downloads are intended.
- Before parallel tests, pre-cache shared assets to avoid xdist download races: `python tests/cache_test_assets.py`; include slow-only assets with `python tests/cache_test_assets.py --slow`.
- Export tests are partitioned by `--export-env`; most shared formats are `base`, while CoreML/TensorFlow/MNN/NCNN/ExecuTorch and isolated vendor exports need their matching env from `ultralytics/engine/exporter.py` and `.github/scripts/create-export-env.py`.

## Formatting And Docs
- Formatting in CI is handled by `ultralytics/actions@main`: Python uses Ruff plus docformatter, and YAML/JSON/Markdown/CSS use Prettier.
- Local docs checks mirror CI with `ruff check --extend-select F,I,D,UP,RUF,FA --target-version py39 --ignore D100,D104,D203,D205,D212,D213,D401,D406,D407,D413,RUF001,RUF002,RUF012 .` and `python docs/build_docs.py`.
- `pyproject.toml` sets a 120-column line length for Ruff, YAPF, isort, and docformatter; Google-style docstrings are expected but many docstring rules are intentionally ignored in docs CI.

## Test And Runtime Gotchas
- `YOLO_AUTOINSTALL=false` is used in CI so missing optional deps fail explicitly instead of installing during tests.
- CI redirects Ultralytics caches with `yolo settings weights_dir=... datasets_dir=...`; local tests may download weights/datasets into the configured global Ultralytics settings directories if you do not override them.
- Tests intentionally exercise paths with spaces via `tests.MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"`; keep shell quoting and path handling robust.
- `tests/conftest.py` seeds tests at session start, removes `slow` tests during collection unless `--slow` is set, and cleans exported artifacts from `WEIGHTS_DIR` at session finish.
- For pytest-xdist export tests, use the `isolated_model` fixture when a test exports from a model file; shared model paths can race and overwrite `.onnx`, `.torchscript`, or export directories.
- CUDA-specific coverage is in `tests/test_cuda.py`; many cases skip without CUDA, and the CI GPU job runs that file separately.
