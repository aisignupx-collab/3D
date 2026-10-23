# Local 3D Asset Pipeline

Prompt or reference image in, engine-ready asset out (GLB / FBX / OBJ), on your own GPU.
The pipeline chains a local LLM, a diffusion model, a feed-forward image-to-3D model and
headless Blender, and closes the loop with a vision-model audit that feeds corrections back
into the Blender stages until the asset passes.

```bash
python main.py --prompt "A functional MP5 submachine gun" --target_format fbx --enable_rigging
python main.py --prompt "Medieval castle wall" --texture_resolution 4096
python main.py --image_input photos/armchair.jpg --target_format glb
python main.py --prompt "a knight character in armor" --enable_rigging   # Rigify biped
```

---

## Architecture

```mermaid
flowchart LR
    P[prompt / image] --> S1
    subgraph GPU["Python + PyTorch/CUDA"]
      S1["1 Concept agent<br/>LLM spec: scale, parts, materials<br/>SDXL/FLUX views + rembg"] --> S2["2 Reconstruction agent<br/>TripoSR / InstantMesh / OpenLRM"]
    end
    subgraph BL["Headless Blender (bpy)"]
      S3["3 Topology<br/>clean, quad remesh,<br/>upright + orient, metric scale"] --> S3b["3b Shape templates<br/>rebuild from box / cylinder /<br/>cone / wedge ... recipes"]
      S3b --> S5a["5 Segmentation cut<br/>(props: parts + pivots)"]
      S5a --> S4["4 Surfacing<br/>shared UVs, PBR bake"]
      S4 --> S5b["5 Rigging<br/>(characters: Rigify / generated)"]
      S5b --> S6["6 Audit renders<br/>front/side/wire/shaded/UV-checker"]
    end
    S2 --> S3
    S6 --> A{"Vision LLM<br/>+ mesh metrics"}
    A -- "fixes (remesh, rescale, re-unwrap, ...)" --> S3
    A -- approved --> E[export GLB/FBX/OBJ]
```

| Stage | Agent / file | What it does |
|---|---|---|
| 1 | `concept_planner.py` | LLM turns the prompt into an `AssetSpec`: category, **real-world dimensions** (MP5 = 680 mm, doors 2.1 m), fit axis, materials (base colour / metallic / roughness), movable parts with normalised regions, pivots and motion limits, rig type, face budget, per-view diffusion prompts. A built-in knowledge base is the offline fallback and sanity-checks LLM scale (rejects answers in centimetres, etc.). |
| 1 | `reference_generator.py` | Diffusers (SD1.5 / SDXL / FLUX) or ComfyUI API generates isometric / front / side views on white; `rembg` isolates the object; images are centred on grey at a fixed foreground ratio (the input format TripoSR/InstantMesh were trained on) and assembled into a concept sheet. Image input skips diffusion. |
| 2 | `mesh_reconstruction.py` | TripoSR in-process on CUDA (vertex colours), InstantMesh / OpenLRM in their own venvs (pinned, conflicting deps), any other CLI model via a command template, or a no-GPU `primitive` placeholder for testing. VRAM is freed between stages. |
| 3 | `blender_automation.py` `op_cleanup` | Merge-by-distance, degenerate/loose removal, floater removal, upright/orientation correction (feed-forward models reconstruct in the camera frame, so a 3/4 view from above comes out yawed ~45° *and* tilted by the camera elevation; a minimum-volume bounding-box search over the convex hull undoes both), grounding, uniform scale to the spec's fit dimension, BVH collision check between shells, then Quadriflow (on a voxel-remeshed manifold base, shrink-wrapped back to the source) / voxel / voxel+shrinkwrap / decimate, hole filling, smoothing, auto-smooth normals. Boolean-unions intersecting shells when not voxelising. |
| 3b | `shape_templates.py` + `op_primitives` | Hard-surface assets are **rebuilt from clean primitives** instead of shipping the lumpy scan: box, cylinder, cone/frustum, sphere, hemisphere, half-cylinder, wedge (gable), pyramid and torus, with bevels, arrays (merlons, planks) and boolean subtraction (doorways, arches). Recipes come from the library (chest, crate, barrel, table, chair, door, castle wall/tower/gate, house, sword, spear, hammer, car, pillar), or the planner LLM composes one from the same vocabulary. With no recipe, every primitive is fitted and the best one is used if it clearly matches (e.g. a crate → box). Component extents snap to the scan, and the fit error decides whether the template replaces the scan (`shapes.mode: auto`). The scan stays in the file as the texture source. |
| 5 | `op_segment` | Props/weapons/vehicles: each part is a **solid boolean cut** (`part = body ∩ box`, `body = body − box`) so both pieces stay watertight with planar mating faces; separate loose shells are split off untouched. Origin set to the pivot, parented to the body, motion limits added as constraints + custom properties (exported as glTF extras / FBX props). Optional `--print_clearance_mm` gap for multi-part 3D printing. Regions come from the planner and are refined by the vision model locating each part on a front render. |
| 4 | `op_texture` | For templates: material IDs come from each component (iron bands → metallic, wood → rough). The scan's colours are baked on, but not its lumpy normals. Smart UV project (or sharp-edge seams / lightmap pack) of body **and** parts into one layout with uniform texel density, then Cycles bakes: albedo from the raw reconstruction's vertex colours / texture (selected-to-active emission bake) or by projecting the reference views, tangent-space normal map from the high-res source, roughness/metallic derived per texel by material-ID against the spec palette. Wired into one `Principled BSDF`. 1K/2K/4K. |
| 5 | `op_rig` | Characters/creatures: Rigify human / quadruped metarig fitted to the mesh and generated, or a generated biped/quadruped skeleton; automatic (bone-heat) weights with an inverse-distance fallback when bone heat fails. |
| 6 | `op_render` + `vision_auditor.py` | Renders front + side (orthographic), clay+wireframe, shaded and UV-checker views. Blender measures watertightness, non-manifold edges, floaters, dimensions vs spec, UV stretch and polycount; the vision LLM judges proportions, missing features, artefacts and texture problems against the concept sheet. Both produce issues with fixes from a closed vocabulary that `apply_fixes` maps onto new Stage 3/4 parameters. The loop repeats until approved or `max_iterations`; the best iteration is exported. |

**Why segmentation runs before surfacing.** Cutting a part creates new faces. Cut after baking,
those faces have no UVs or texture and each piece would need its own re-bake. Cutting first lets
body and parts share one UV layout and one texture set. Rigging changes no geometry, so it runs
after Stage 4.

**Why metrics and vision are combined.** A vision model can't tell from pixels whether a mesh is
manifold. Blender measures that exactly. The vision model is used for what only it can judge:
whether the object looks right.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA 8 GB (SD1.5/SDXL-turbo + TripoSR, `enable_cpu_offload: true`) | 16-24 GB (SDXL/FLUX-schnell + TripoSR, or InstantMesh) |
| Driver | NVIDIA 535+ (CUDA 12.x) | latest Studio/Game Ready |
| RAM | 16 GB | 32 GB+ (FLUX offload, 4K bakes) |
| Disk | 25 GB for models | 60 GB with FLUX + InstantMesh |
| Blender | 4.2 LTS | 4.2 LTS (tested) |
| Python | 3.10 | 3.11 |

Approximate VRAM per stage (stages run sequentially and unload in between, so the peak is the largest one):
SDXL fp16 ~8 GB · FLUX.1 bf16 with CPU offload ~12 GB · TripoSR ~6 GB at `mc_resolution: 256` ·
InstantMesh large ~16 GB · Qwen2.5-VL-7B in Ollama ~6 GB (served separately, so count it too unless it runs on another GPU or the CPU).

---

## Installation

### 1. NVIDIA driver and CUDA

* Install a current NVIDIA driver (`nvidia-smi` should report CUDA 12.x).
* PyTorch wheels ship their own CUDA runtime; you only need the **CUDA toolkit** (`nvcc`) to
  build `torchmcubes` (TripoSR) or InstantMesh's extensions. Its major/minor version must match
  the wheel (`python -c "import torch; print(torch.version.cuda)"`).
  * Linux: `sudo apt install cuda-toolkit-12-4` (NVIDIA repo) or `conda install -c nvidia cuda-toolkit=12.4`
  * Windows: CUDA Toolkit 12.4 installer + Visual Studio 2022 Build Tools (C++ workload)
* If the build can't detect your GPU, set the architecture explicitly, e.g.
  `export TORCH_CUDA_ARCH_LIST="8.6"` (RTX 30xx), `"8.9"` (RTX 40xx), `"12.0"` (RTX 50xx).

### 2. Python environment and PyTorch

```bash
conda create -n asset3d python=3.11 -y && conda activate asset3d
# Install PyTorch FIRST, from the CUDA index matching your driver/toolkit:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -r requirements.txt
pip install "rembg[gpu]"          # swaps in onnxruntime-gpu for faster background removal
```

### 3. Image-to-3D backend

**TripoSR (default, in-process):**
```bash
git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR
# torchmcubes compiles against the installed torch, so build isolation must be off:
pip install scikit-build-core cmake ninja pybind11
pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
```
Weights (`stabilityai/TripoSR`, ~1.7 GB) download from Hugging Face on first run.

**InstantMesh (optional, own venv):**
```bash
git clone https://github.com/TencentARC/InstantMesh third_party/InstantMesh
cd third_party/InstantMesh && python3.10 -m venv .venv && . .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7 && pip install -r requirements.txt
```
Then set `reconstruction.backend: instantmesh` and `instantmesh_python: third_party/InstantMesh/.venv/bin/python`.
OpenLRM works the same way (`openlrm_repo`, `openlrm_python`). For other models (TRELLIS, Hunyuan3D,
SF3D), set `backend: command` and a `command_template` such as
`{python} /path/to/run.py --image {input_rgba} --out {output_dir}`. The newest mesh written to `{output_dir}` is used.

### 4. Diffusion model

* **Diffusers (default):** SDXL base (~7 GB) downloads on first use. For FLUX:
  `huggingface-cli login` (FLUX.1-dev is gated), then
  `diffusion.model_id: black-forest-labs/FLUX.1-schnell`, `steps: 4`, `guidance_scale: 0.0`.
* **ComfyUI:** start it with `python main.py --listen`, put the checkpoint in `models/checkpoints`,
  set `diffusion.backend: comfyui`. A custom graph can be supplied as API-format JSON
  (*Save (API Format)* in ComfyUI) with `{{prompt}}`, `{{negative}}`, `{{seed}}`, `{{width}}`, `{{height}}` placeholders.

### 5. LLM and vision model

**Ollama (default):**
```bash
ollama pull qwen2.5:14b-instruct    # concept planner
ollama pull qwen2.5vl:7b            # vision auditor (alternatives: llava:13b, minicpm-v)
```
**vLLM / LM Studio / llama.cpp server:** set `provider: openai_compatible` and `base_url`
(`http://localhost:8000` for `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`, `http://localhost:1234` for LM Studio).
**Cloud:** `provider: anthropic` (model e.g. `claude-sonnet-5-5`, key in `ANTHROPIC_API_KEY`) or
`provider: openai` (`OPENAI_API_KEY`).
Without any LLM the pipeline still runs: planning falls back to the knowledge base and the audit uses
mesh metrics only (`--offline`).

### 6. Blender

Install **Blender 4.2 LTS** (<https://www.blender.org/download/lts/>). On Linux the portable tarball is enough:
```bash
wget https://download.blender.org/release/Blender4.2/blender-4.2.9-linux-x64.tar.xz
tar xf blender-4.2.9-linux-x64.tar.xz -C /opt && export BLENDER_PATH=/opt/blender-4.2.9-linux-x64/blender
```
The executable is found from (in order) `--blender_path`, `blender.executable` in `config.yaml`,
`$BLENDER_PATH`, `PATH`, then the default install locations:
* Windows: `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe`
* macOS: `/Applications/Blender.app/Contents/MacOS/Blender`

Blender runs with `--background --factory-startup`, so your personal add-ons and preferences don't
affect results. Cycles uses OptiX/CUDA/HIP automatically when `blender.use_gpu: true`. Audit renders
default to Cycles because it works headless on any machine. EEVEE (`render.engine: eevee`) needs a
GPU-backed OpenGL/EGL context.

### 7. Verify

```bash
cp config.example.yaml config.yaml          # then edit
python main.py --doctor                     # CUDA/VRAM, packages, TripoSR, Blender, LLM endpoints
python main.py --dry_run --prompt "A functional MP5 submachine gun" --enable_rigging   # Blender stages only, no GPU
pytest                                       # unit tests; Blender tests run when Blender is found
```

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `torchmcubes must be built against an installed PyTorch` | Build without isolation: `pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git` (needs `cmake`, `ninja`, `pybind11`, `scikit-build-core` and, for GPU marching cubes, `nvcc`). |
| TripoSR: `Missing key(s) in state_dict: image_tokenizer.model.layers.0...` | `transformers` 5.x renamed the ViT modules. Use `transformers<5` (pinned in `requirements.txt`). |
| `diffusers ... requires huggingface-hub>=1.x` conflict | diffusers ≥ 0.36 and transformers 4.x can't share an env. Use `diffusers<0.36` (pinned). |
| `CUDA out of memory` in Stage 1 | `diffusion.enable_cpu_offload: true`, or an SDXL-turbo / FLUX-schnell model. Make sure Ollama isn't holding the vision model on the same GPU (`OLLAMA_KEEP_ALIVE=0`). |
| Asset lies on its side / faces backwards | Auto-orientation fixes tilts up to `cleanup.max_tilt_correction` (45°) but can't tell which face is the front. For a backend with a different up axis, set `cleanup.import_rotation_deg` (e.g. `[90, 0, 0]`). The audit can also request a `reorient` fix. |
| Bone heat weighting fails | Happens on non-manifold or self-intersecting meshes. The rig step falls back to proximity weights automatically; `--remesh_mode voxel` gives bone heat a clean manifold. |
| Blender crash log in `/tmp/*.crash.txt` | Check `blender_jobs/blender_<op>.log` in the run folder. Every op writes its full log there. |

---

## Usage

| Flag | Meaning |
|---|---|
| `--prompt TEXT` | asset description (Stage 1 generates references) |
| `--image_input PATH` | use this image instead of diffusion (prompt optional; the VLM names the object) |
| `--target_format glb\|fbx\|obj` | export format (textures embedded in GLB/FBX, copied next to OBJ) |
| `--enable_rigging` | Stage 5: part segmentation for props, armature + weights for characters/creatures |
| `--rig_method rigify\|generated` | Rigify metarig, or a lightweight generated skeleton (19 bones biped) |
| `--print_clearance_mm N` | cut an N mm gap between segmented parts (3D printing) |
| `--texture_resolution 1024\|2048\|4096` | bake resolution |
| `--remesh_mode`, `--target_faces` | Stage 3 topology overrides |
| `--shape_mode auto\|primitives\|mesh` | rebuild from primitive templates when they fit the scan (default) / always / never |
| `--recon_backend`, `--diffusion_backend` | swap models |
| `--max_iterations N` | audit/refinement rounds (default 3) |
| `--resume RUN_DIR` | re-run Stages 3-6 on an existing run (reuses the spec, references and raw mesh) |
| `--offline` | no LLM/VLM calls |
| `--dry_run` | offline + placeholder mesh + no diffusion: exercises Blender end to end without a GPU |
| `--doctor` | environment check |

Every run writes a self-contained folder:

```
output/a_functional_mp5_submachine_gun_20261023_101500/
├── manifest.json          # options, config, spec, every iteration's params/metrics/audit, timings
├── spec.json              # AssetSpec from Stage 1
├── references/            # *_raw / *_rgba / *_prepared.png, concept_sheet.png
├── raw/                   # Stage 2 mesh
├── iter_1/                # stage3_topology.blend, stage5_segmented.blend, stage4_surfacing.blend, textures/
├── renders/iter_1/        # front, side, wireframe, shaded, uv_checker
├── blender_jobs/          # job/result JSON + full Blender log per operation
└── export/                # <name>.fbx|glb|obj, <name>.blend, previews/
```

### Primitive templates

A crate should be a perfect box and a tower a true cylinder with a cone roof, not whatever blob the
reconstruction produced. Stage 3b handles this:

1. **Pick a recipe.** Planner LLM shapes win, then the library match for the prompt. Organic
   assets (characters, creatures) are never templated.
2. **Fit.** Each component's free sides snap to the scan's points (shared sides and the outer bbox
   stay fixed, so pieces keep meeting cleanly). The composite is scored by the symmetric surface
   distance to the scan.
3. **Decide.** In `auto` mode, a recipe is used when its error is below `shapes.recipe_max_error`.
   A single-shape guess must beat the stricter `classify_max_error`. Otherwise the cleaned scan is
   kept (e.g. an MP5 whose detail no simple recipe captures).
4. **Build.** Components are boolean-unioned into one watertight shell. `subtract` components cut
   openings, and components with a `part` become separate objects with the spec pivot and motion
   limits (chest lid, door leaf, car wheels).
5. **Texture.** Stage 4 bakes the scan's colours onto the clean shapes, with the bake cage widened
   to the measured template-to-scan distance. Each component's `material` drives the metallic and
   roughness maps.

Recipe syntax (`asset_pipeline/shape_templates.py`). Coordinates are fractions of the asset's
bounding box (X = length, Y = depth, Z = height):

```python
{"keywords": ["tower", "turret"], "shapes": [
    C("tower", "cylinder", (0.06, 0.06, 0), (0.94, 0.94, 0.72), axis="Z", material="stone"),
    C("roof", "cone", (0, 0, 0.74), (1, 1, 1), axis="+Z", material="roof"),
    C("door", "half_cylinder", (0.4, -0.05, 0), (0.6, 0.1, 0.16), axis="Y", operation="subtract"),
]}
```

Add an entry to `RECIPES` to teach the pipeline a new asset family. The planner LLM receives the
same primitive vocabulary and returns `shapes` in this format for anything not in the library.

### The refinement loop

| Fix action | Triggered by | Parameter change |
|---|---|---|
| `make_manifold` | boundary / non-manifold edges | decimate→voxel remesh, else finer voxels + hole fill |
| `increase_detail` | VLM: blobby, lost features | voxel resolution ×1.3, face budget ×1.5 |
| `reduce_noise` | VLM: lumpy / stair-stepped | +2 smoothing iterations |
| `remove_floaters` | small islands | raise minimum island size |
| `rescale_axis` | extent >25 % off spec, or VLM | per-axis scale (bounded 0.7-1.3) |
| `reorient` | VLM: upside down / on its side | extra import rotation |
| `reduce_faces` | >1.5× face budget | target faces ×0.7 |
| `reunwrap` | UV area distortion > 0.5, VLM checker stretching | smaller smart-UV angle → lightmap pack |
| `rebake` | VLM: smearing, black patches | wider cage, larger margin |

If only texture parameters changed, Stage 3 and the segmentation cut are reused and only Stage 4 onward re-runs.

---

## Limitations and tuning notes

* **Reconstruction quality sets the ceiling.** Feed-forward models produce soft, blobby hard-surface
  detail. Stage 3 makes the geometry clean and engine-ready, but it can't invent detail that isn't
  there. For hard-surface hero assets, use InstantMesh or a stronger model through the `command`
  backend, and raise `mc_resolution`.
* **Per-view diffusion images are not multi-view consistent.** Front and side are separate
  generations with a shared seed. Only the primary (isometric) view drives reconstruction. The other
  views serve as audit references and as optional projection textures (`albedo_source: projection`),
  which assume the views roughly line up with the reconstructed silhouette. InstantMesh produces its
  own consistent views internally through Zero123++.
* **Templates trade detail for cleanliness.** A template carries the scan's colours but not its
  carved relief (set `shapes.bake_normal: true` to bake that back as a normal map, lumps included).
  On symmetric shapes the template can't tell which face is the front, so a scan's lock or door
  texture can land on a side face. Templates take the scan's proportions, and the audit flags
  deviations from the spec.
* **Segmentation is region-based.** It works well for parts that occupy a separable box (magazines,
  wheels, doors, lids, slides). Interlocking internals such as bolts inside receivers need manual
  work. Parts whose region lies wholly inside the solid are skipped and reported.
* **Rigify fitting is proportional, not landmark-based.** The metarig is scaled to the mesh. Expect
  to adjust joints on stylised proportions, or use `--rig_method generated` for a minimal game
  skeleton. Generating characters in A-pose or T-pose (the default prompt asks for A-pose) matters
  most.
* **Roughness and metallic maps are derived, not learned.** Each texel is classified against the
  planner's material palette. Swap in a material-estimation model by replacing `derive_pbr` in
  `blender_automation.py`.
* Version-specific Blender API differences (auto smooth, EEVEE id, importers/exporters) go through
  compatibility helpers, and `call_op` drops operator arguments a given version doesn't support.

### Verification status

| Component | How it was verified |
|---|---|
| Primitive templates (12 recipes, auto-classification, parts, material-ID bake) | Real Blender runs on the placeholder and on the real TripoSR chest scan |
| Blender Stages 3-6 + export (all remesh modes, segmentation, Rigify / generated rigs, GLB/FBX/OBJ) | Real headless runs on **Blender 4.2 LTS and 4.5 LTS** (`pytest -m blender` plus end-to-end `--dry_run`) |
| Diffusers → rembg → TripoSR → Blender → export | Real end-to-end run on **CPU** (`sd-turbo` 1-step, TripoSR at `mc_resolution: 160`), including `--resume` |
| Closed loop | Iteration 1 rejected as non-watertight (`--remesh_mode decimate`), fix switched to voxel remeshing, iteration 2 approved |
| Ollama / OpenAI-compatible / Anthropic clients | Request/response wire formats tested against a local fake server, not a live model |
| CUDA GPU paths, SDXL/FLUX, ComfyUI, InstantMesh, OpenLRM | Implemented against their documented interfaces but not executed in this environment; run `python main.py --doctor` first |

## Project layout

```
main.py                         CLI
config.example.yaml             all settings with comments
asset_pipeline/
  config.py                     dataclass config, YAML + env loading, Blender discovery
  llm_client.py                 Ollama / OpenAI-compatible / OpenAI / Anthropic, text + vision
  concept_planner.py            Stage 1 spec agent + knowledge base
  reference_generator.py        Stage 1 diffusion (diffusers/ComfyUI) + rembg preprocessing
  mesh_reconstruction.py        Stage 2 backends
  shape_templates.py            Stage 3b primitive vocabulary + recipe library
  blender_automation.py         Stages 3-6 bpy operations + host-side BlenderRunner
  vision_auditor.py             Stage 6 metric/VLM audit, part localisation, fix mapping
  orchestrator.py               stage agents, refinement loop, manifest/resume
  doctor.py                     environment diagnostics
tests/                          unit tests + real-Blender integration tests
```
