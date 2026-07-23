# VideoClean — Implementation Plan

## 0. Before Writing Code

**Environment setup checklist:**

1. Install Python 3.11+
2. Install `uv` for dependency management: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install FFmpeg system-wide (`brew install ffmpeg` / `apt install ffmpeg`) — verify with `ffmpeg -version` and `ffprobe -version`
4. Initialize repo with the project structure (see Section 8 of the original build plan)
5. Create `pyproject.toml` with initial dependencies: `typer`, `rich`, `opencv-python`, `numpy`, `ffmpeg-python` (or a subprocess wrapper), `torch` (CPU wheel first)

**Decision to lock in before Phase 1:**
Pick the frame-extraction strategy for MVP (temp PNG/JPEG files — simplest) and the inpainting model for MVP. Use a single, well-supported local image-inpainting approach — e.g. OpenCV's `cv2.inpaint` — as a **placeholder backend**, so the full pipeline can be validated end-to-end before integrating a heavier AI model. This unblocks Phase 4/5 without waiting on model research.

---

## Phase 1 — CLI Skeleton

**Goal:** `videoclean info` works; `videoclean remove` exists but is a stub.

**Tasks:**

- Scaffold `src/videoclean/` per the recommended project tree
- Implement `cli.py` with Typer commands: `info`, `remove` (stub), `preview` (stub)
- Implement `exceptions.py` (custom exception hierarchy)
- Implement `utils/logging.py` with `--verbose` flag (Rich-based)
- Implement `Region` parsing/validation (`x1,y1,x2,y2` string → `Region` dataclass) with unit tests

**Acceptance criteria:**

- `videoclean info --help` runs without error
- Region-parsing unit tests pass

---

## Phase 2 — FFmpeg Integration

**Goal:** Round-trip a video with no AI processing: extract frames → reassemble → restore audio → valid output.

**Tasks:**

- `video/ffmpeg.py`: wrapper to run ffmpeg/ffprobe as subprocesses, raising `FFmpegNotFoundError` / `InvalidVideoError` on failure
- `video/metadata.py`: `VideoMetadata` extraction via ffprobe (JSON output parsing)
- `video/extractor.py`: frame extraction to a temp directory
- `video/encoder.py`: frame reassembly to H.264/libx264
- Audio extraction + final mux logic
- Temp directory lifecycle: create under `/tmp/videoclean/job-<uuid>/`, cleanup on success, `--keep-temp` flag to preserve on failure/debug

**Acceptance criteria:**

- Feeding a video through extract → encode → mux with **no mask applied** produces an output matching original duration, resolution, and audio, verified by an integration test

---

## Phase 3 — Manual Masking

**Goal:** User can specify and visually confirm a watermark region.

**Tasks:**

- `masks/manual.py`: generate binary mask from `Region` (0 = keep, 255 = reconstruct), with configurable `--mask-padding`
- Wire `--region X1,Y1,X2,Y2` into `remove` command with validation (region inside video bounds, x1<x2, y1<y2)
- Implement `videoclean preview` command: render original frame + region box + mask overlay to an image file

**Acceptance criteria:**

- `videoclean preview input.mp4 --region ... --output preview.png` produces a correct overlay image
- Invalid regions produce a clear, human-readable error (not a raw stack trace)

---

## Phase 4 — Inpainting Model

**Goal:** Abstract inpainting interface with a working placeholder model.

**Tasks:**

- `inpainting/base.py`: define `InpaintingEngine` ABC (`load`, `process_frame`, `process_video`)
- `inpainting/frame_inpainter.py`: implement `FrameInpaintingEngine` using `cv2.inpaint` (or a chosen open-source model) as MVP backend
- `models/registry.py`: simple registry so `--model` can select between implementations later

**Acceptance criteria:**

- Given a frame + mask, `process_frame` returns a plausible reconstructed frame
- Model logic has zero dependency on CLI or pipeline code (verified by import structure)

---

## Phase 5 — End-to-End Pipeline

**Goal:** Fully functional MVP matching the Definition of Done.

**Tasks:**

- `core/pipeline.py`: orchestrate the 17-step processing pipeline (validate → metadata → region → device → temp dir → extract → mask → load model → process → encode → mux → cleanup)
- `core/job.py`: `ProcessingConfig` + job state tracking
- Wire `remove` command fully into the pipeline
- Add Rich progress bars per stage (not per-frame)
- Add non-zero exit codes on failure

**Acceptance criteria:**

- `videoclean remove input.mp4 --output cleaned.mp4 --region 20,20,300,120` runs end-to-end and produces a playable video preserving duration, resolution, and audio sync
- All 14 MVP requirements (Section 28 of build plan) are satisfied

---

## Phase 6 — GPU Support

**Goal:** Automatic device selection with CPU fallback.

**Tasks:**

- `utils/device.py`: detect CUDA / MPS / CPU, implement `auto` selection logic
- Wire `--device` CLI flag through to pipeline and model loading
- Display selected device before processing begins

**Acceptance criteria:**

- Runs correctly on CPU-only machines
- Correctly selects CUDA or MPS when available and requested/auto-detected

---

## Phase 7 — Temporal Consistency

**Goal:** Reduce flickering vs. frame-by-frame MVP approach.

**Tasks:**

- `inpainting/video_inpainter.py`: implement `TemporalVideoInpaintingEngine` using neighboring-frame context (optical flow, feature propagation, or a temporal-aware model)
- Update pipeline to support chunked/windowed frame processing instead of strict per-frame

**Acceptance criteria:**

- Side-by-side comparison vs. Phase 4 output shows reduced flicker/artifacting on a static-background test video

---

## Phase 8 — Moving Watermark Support

**Goal:** Track and remove watermarks that move across frames.

**Tasks:**

- `masks/tracker.py`: implement tracking (OpenCV object tracker or optical-flow-based) from an initial user-supplied region
- Generate per-frame masks instead of one static mask
- Add `--track` CLI flag

**Acceptance criteria:**

- `videoclean remove input.mp4 --track --region ...` correctly follows a moving watermark through a test video

---

## Phase 9 — Automatic Detection

**Goal:** Suggest watermark regions without requiring manual input.

**Tasks:**

- `masks/detector.py`: detect persistent/static regions across sampled frames
- Implement `videoclean detect input.mp4` to output candidate regions
- Require explicit user confirmation before applying detected regions to `remove`

**Acceptance criteria:**

- `videoclean detect` produces reasonable candidate region(s) on test videos with an obvious static logo/watermark
- No automatic modification occurs without user confirmation

---

## Testing Strategy (ongoing, every phase)

- **Unit tests:** region parsing/validation, mask generation, metadata parsing, device detection, config loading
- **Integration tests:** small test videos through extract → process → encode → mux → validate (check output exists, is playable, correct duration/resolution/audio)
- **Quality tests (manual):** test videos with static watermark, semi-transparent watermark, text watermark, logo watermark, over simple vs. complex backgrounds

Build a small `tests/fixtures/` set of short sample videos early (Phase 2) so every later phase can reuse them.

---

## Cross-Cutting Rules (apply throughout)

1. MVP first — do not build automatic detection before manual region processing is solid.
2. Keep the CLI decoupled from the AI model (interface in `inpainting/base.py` is the only contract).
3. Isolate FFmpeg calls in `video/ffmpeg.py` only.
4. Type hints everywhere.
5. No entire-video-in-RAM loads.
6. Robust temp file handling — always clean up unless `--keep-temp`.
7. macOS/Linux first; Windows support after core pipeline is stable.
8. No network calls during normal processing (model downloads are a separate, explicit setup step).
9. Human-readable errors, never raw stack traces, for user-facing failures.

---

## Suggested Order of Work for a Single Contributor

1. Phase 1 → 2 → 3 (get a working non-AI round-trip with masking/preview)
2. Phase 4 → 5 (bolt on placeholder inpainting, reach MVP Definition of Done)
3. Phase 6 (GPU support — relatively isolated, can be done anytime after Phase 5)
4. Phase 7 → 8 → 9 (quality and feature improvements, in that order)
