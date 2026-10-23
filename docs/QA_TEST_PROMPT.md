# QA test brief: local 3D asset pipeline

> Paste everything below the line into the testing AI (Claude Code, Codex, OpenCode or similar) running
> on the target PC with shell access. It returns a written audit plus a results zip, which go back to
> the developer.

---

You are an independent QA engineer and technical-art reviewer. Install, run and stress-test a local
AI 3D-asset pipeline on this machine, then write a complete, evidence-based audit of what works,
what fails, what looks wrong, and how you would test it further. You are **testing, not
developing**. Your audit goes back to the pipeline's developer, who will fix the issues.

## 0. Ground rules

1. **Evidence over opinion.** Every claim needs the exact command, its relevant output (trimmed, but
   never paraphrased), and file paths. Mark anything you did not observe directly as
   `SPECULATION:`.
2. **Don't fix the code to make tests pass.** If a bug blocks you, you may apply the smallest
   possible workaround to continue. Record it in the report as a bug, and save the change with
   `git diff > qa_patches.diff`. Never commit or push to the repository.
3. **Secrets:** API keys go in environment variables only. Never write them into files, logs or the
   report. Redact them if they show up in output.
4. **Be honest about scope.** If you skip something (no API key, no Unreal, ran out of time), list it
   under "Not tested" with the reason. A short honest report beats a long padded one.
5. Time every run (wall clock) and record GPU memory (see §4.1).
6. Keep going after failures. One broken stage must not stop the rest of the matrix. Use `--resume`
   or other assets.
7. Work in a fresh folder. If anything is ambiguous, pick the option a typical user would pick,
   note the choice, and continue. Don't stop to ask.

## 1. What you are testing

Repository: `https://github.com/aisignupx-collab/3D`, branch **`claude/modest-mccarthy-gk38in`**.
Read `README.md` fully before starting. It is also under test: report every place where it is wrong,
unclear or incomplete.

The pipeline (`main.py`) turns a text prompt or an image into a game-ready 3D asset
(GLB/FBX/OBJ):

| Stage | What it does | Main code |
|---|---|---|
| 1 | LLM plans the asset (real-world size, materials, moving parts, rig type); diffusion renders reference views; rembg cuts them out | `asset_pipeline/concept_planner.py`, `reference_generator.py` |
| 2 | Image-to-3D (TripoSR by default) | `mesh_reconstruction.py` |
| 3 | Blender cleanup: upright/orientation fix, metric scaling, quad remesh | `blender_automation.py` `op_cleanup` |
| 3b | Rebuild hard-surface assets from primitive templates (box, cylinder, cone ...) or a three-view "blueprint" hull for guns/tools | `shape_templates.py`, `op_primitives` |
| 4 | UVs and PBR textures: reference image projected back on, mirror fill, baked colours, material IDs | `op_texture` |
| 5 | Props: moving parts cut out with pivots (lid, magazine, wheels). Characters: joints detected from the mesh, Rigify or a 21-bone skeleton, skin weights, pose test | `op_segment`, `op_rig` |
| 6 | Audit renders, mesh metrics and a vision-LLM review, with automatic fixes fed back for up to 3 iterations | `op_render`, `vision_auditor.py`, `orchestrator.py` |

Integrations: Ollama / LM Studio / vLLM / OpenAI / Anthropic / DeepSeek / Gemini / OpenRouter
providers, a `cli` provider (an agent CLI as the model), and `mcp_server.py` (MCP tools for Claude
Code / Codex / OpenCode).

Helper scripts for you:
- `scripts/inspect_export.py` (run inside Blender) prints what an engine sees in an exported file:
  objects, real dimensions, faces, UVs, textures, bones, skin-weight sanity and warnings.
- `scripts/collect_results.py` bundles all runs into `qa_results/` + `qa_results.zip` (§9).

## 2. Phase A: environment inventory

Record:
- OS and version, CPU (model, cores), RAM, free disk space.
- GPU model, VRAM and driver (`nvidia-smi`); CUDA toolkit (`nvcc --version`, if installed).
- Python versions available, conda/venv tooling.
- Whether Blender, Ollama, Git, ComfyUI, Unity/Unreal/Godot are installed, and their versions.

## 3. Phase B: installation (follow the README literally)

1. Clone the branch into a fresh folder and follow `README.md` → *Installation* **exactly as
   written**, in order.
2. Log every command and how it went. For each problem, record:
   - the exact error;
   - what you did to get past it;
   - whether the README should change (quote the line, propose the fix).
3. Install TripoSR and torchmcubes as documented. Note whether torchmcubes built with CUDA support
   (`python -c "import torchmcubes, torch; print(torch.cuda.is_available())"`).
4. Install Blender 4.2 LTS (or report which version you used and why).
5. Install Ollama and pull the two default models from the README.
6. `pip install "mcp>=1.2"` for the MCP tests.
7. Copy `config.example.yaml` to `config.yaml`. Then also try `--config configs/rtx4090.yaml` on a
   24 GB GPU.
8. Run `python main.py --doctor` and paste the full output.

Report the total install time and a 1-10 "how painful was it" score with reasons.

## 4. Phase C: automated and smoke tests

```bash
pytest -q                      # all tests; Blender tests run when Blender is found
pytest -q -m blender -rA       # integration tests only, with per-test results
python main.py --dry_run --prompt "A functional MP5 submachine gun" --enable_rigging --target_format fbx
python main.py --dry_run --prompt "a knight character in armor" --enable_rigging
```

Paste the pass/fail counts and any failure output in full.

### 4.1 Measuring time and VRAM

For every GPU run in §5, log GPU memory in parallel and report the peak:
- Linux: `nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > vram_<ID>.csv`
- Windows: the same command in a second terminal.

Also record each run's `timings_s` (printed at the end of every run and stored in `manifest.json`).

## 5. Phase D: the GPU test matrix

Use the default `config.yaml` (Ollama local models) unless the row says otherwise. Run every row.
Use `--output_dir qa_output` for all of them.

| ID | Command (append `--output_dir qa_output`) | What to check |
|---|---|---|
| T01 | `--prompt "a wooden crate"` | Becomes a clean bevelled box? Texture sharp on the visible faces? |
| T02 | `--prompt "a wooden treasure chest with iron bands" --enable_rigging --target_format glb` | Box template with iron bands; lid is a separate object hinged at the back (rotate it); iron reads metallic, wood rough |
| T03 | `--prompt "a wooden barrel"` | Cylinder with two hoops; dimensions ~0.6 × 0.6 × 0.9 m |
| T04 | `--prompt "a medieval castle tower"` | Cylinder + cone roof + door recess; ~6 × 6 × 16 m |
| T05 | `--prompt "medieval castle wall"` | Wall + walkway + merlons; ~10 × 2.5 × 8 m |
| T06 | `--prompt "a castle gate"` | Two towers, roofs, gatehouse with an arched passage cut through |
| T07 | `--prompt "A functional MP5 submachine gun" --enable_rigging --target_format fbx` | Blueprint rebuild (straight edges)? ~0.68 m long? Magazine / trigger / cocking-handle parts and pivots |
| T08 | `--prompt "a claw hammer"` | Handle + head; ~0.33 m |
| T09 | `--prompt "a longsword"` | Blade, crossguard, grip, pommel; ~1.0 m |
| T10 | `--prompt "a red sedan car" --enable_rigging` | Body + cabin; 4 wheels as separate parts with pivots at the hubs |
| T11 | `--prompt "a knight character in armor, full body" --enable_rigging --target_format fbx` | Rigify rig; joints in the right places; pose test numbers; posed renders |
| T12 | `--prompt "a knight character in armor, full body" --enable_rigging --rig_method generated --target_format glb` | 21-bone game skeleton; same checks as T11 |
| T13 | `--prompt "a wolf" --enable_rigging` | Quadruped skeleton; legs land in the actual legs; head direction |
| T14 | `--image_input <a real photo of a chair, mug or similar> --target_format glb` | Image path: rembg cutout quality, projection of the photo onto the model |
| T15 | `--prompt "a sci-fi surveillance drone"` (not in the recipe library) | Does the planner LLM compose a primitive recipe, or does the pipeline fall back? Result quality |
| T16 | `--prompt "a wooden treasure chest" --shape_mode mesh` | Same asset without templates, to compare scan vs template quality against T02 |
| T17 | `--prompt "a stone pillar" --texture_resolution 4096` | 4K bake time and VRAM |
| T18 | `--resume qa_output/<T07 run dir> --target_format obj` | Resume works without re-running the GPU stages; OBJ comes with its textures |

For **each** run, record:
- success or failure and wall-clock time; `timings_s` per stage; peak VRAM;
- the path the asset took (`recipe`, `classified:<shape>`, `blueprint`, or `scan mesh`) and its fit
  error (in `manifest.json` → `iterations[].stage3b`);
- texture projection: `silhouette_iou`, `projected_fraction`, or the reason it was skipped
  (`stage4.projection`);
- rigs: detected landmarks and fallbacks, weights method, `pose_test.bad_edge_fraction`;
- the audit: approved, score, every issue, the fixes applied between iterations, and whether the
  fixes actually improved the next iteration;
- the output of `blender --background --factory-startup --python scripts/inspect_export.py -- <exported file>`.

## 6. Phase E: look at every asset like a technical artist

Open each export (Blender, or a glTF viewer such as https://gltf-viewer.donmccurdy.com for GLB) and
look at `export/previews/` and `renders/iter_*/`. Score each asset from 1 to 5 on the following,
with one line of justification per score:

| Criterion | What "5" means |
|---|---|
| Shape fidelity | Recognisably the requested object; proportions right; no missing major parts |
| Geometry cleanliness | Straight edges where they should be straight, no lumps, holes, spikes or floating debris |
| Topology | Sensible face count for a game asset; quads where expected; no shading artefacts |
| Real-world scale | Measured dimensions match reality (use `inspect_export.py` numbers); rests on the ground; faces −Y (front) |
| Texture | Sharp where the reference was visible; aligned with the geometry; no stretching, smearing, black patches or obvious seams; mirrored-detail duplicates noted |
| Materials | Metal reads as metal, wood/stone/cloth as non-metal; roughness plausible |
| Moving parts (if any) | Separate objects, pivot at the real hinge/axis, motion feels right when rotated or translated |
| Rig (characters) | Bones inside the limbs at the right joints; rotate shoulders, elbows, hips and knees by 45-90° and report tearing, candy-wrapping or collapsing |
| Engine readiness | Imports cleanly into a game engine if available (Unity/Unreal/Godot): scale, orientation, materials, skeleton |

Attach the screenshots you based the scores on, or reference the renders by path.

## 7. Phase F: AI models and integrations

1. **Local:** confirm the default Ollama planner and vision models are used (not the offline
   fallback). The run log should not say "unreachable". Check that the audit reports contain a
   `vision` section.
2. **Cloud** (only with keys the owner provides through environment variables): rerun T02 and T07
   with `llm: {provider: deepseek, model: deepseek-chat}` and
   `vision: {provider: gemini, model: gemini-2.5-flash}`, or OpenRouter equivalents. Compare plan
   quality, audit quality, speed and errors against local.
3. **`cli` provider:** if Claude Code, Codex or OpenCode is installed, set
   `llm: {provider: cli, command: "<agent> ... {prompt}"}` and run T01. Report whether the JSON
   came back parseable and how long it took.
4. **MCP:** register `mcp_server.py` with an agent (`claude mcp add asset-pipeline -- python <repo>/mcp_server.py`,
   or the Codex/OpenCode equivalent from the README). From the agent, call `doctor`,
   `list_shape_templates`, `generate_asset` (prompt "a wooden barrel") and `list_runs`. Report what
   worked, timeouts, and how clear the tool descriptions were to the agent.

## 8. Phase G: robustness and edge cases

Try each one and report the behaviour (graceful error vs crash vs silent wrong result):
- Blender path wrong or missing; Ollama stopped mid-run; no internet (models already cached).
- Extreme scales: `"a coin"`, `"a skyscraper"`. Ambiguous prompts: `"a thing"`, an empty prompt,
  a 500-word prompt, non-English (`"una espada medieval"`).
- An image with a busy background, a very small object, or a transparent PNG.
- Interrupting a run (Ctrl+C during Stage 4), then `--resume` it.
- Running two pipelines at once (VRAM contention).
- Every `--target_format` for one asset: do GLB/FBX/OBJ agree on dimensions and parts?
- `--max_iterations 1` vs `5`: does the loop converge or oscillate?

## 9. Phase H: package the results

```bash
python scripts/collect_results.py --runs qa_output output --out qa_results --notes QA_REPORT.md
```

Send back `QA_REPORT.md` **and** `qa_results.zip`, plus `qa_patches.diff` if you had to patch anything.

## 10. Your own test design

The developer wants your judgement, not only the checklist. Add a section: **"How I would test
this properly."** It should cover:
- what the current automated tests miss (read `tests/`), and five concrete tests you would add;
- a golden set of assets with acceptance criteria you would use for regression testing;
- the metrics you would trust and the ones you would not (e.g. is `fit_error` meaningful, is the
  vision score consistent between runs?);
- how you would test texture alignment, rig deformation and engine import automatically.

## 11. Report format (`QA_REPORT.md`)

Use exactly these headings:

```
# QA report: 3D asset pipeline (<date>, <tester name/model>)

## 1. Executive summary
5-10 bullets: overall verdict, the 3 most serious problems, what impressed you, ready-for-use? (yes / with caveats / no)

## 2. Environment
(inventory from Phase A)

## 3. Installation log
Numbered steps, problems hit, workarounds, README fixes. Install time and pain score.

## 4. Automated tests
pytest / dry-run results, failures with output.

## 5. Test matrix results
One table row per test ID: status | time | peak VRAM | path taken | audit score/approved | key numbers.
Then one subsection per test with details and inspect_export output.

## 6. Asset quality scorecard
Table of the §6 scores per asset, with one-line justifications and screenshot paths.

## 7. AI model and integration results
Local, cloud, cli, MCP.

## 8. Robustness / edge cases
One line per case: behaviour, verdict.

## 9. Bugs
One entry per bug:
### BUG-<n>: <title>
- Severity: critical | high | medium | low
- Area: install | stage 1..6 | export | integrations | docs
- Steps to reproduce: (exact commands)
- Expected / actual
- Evidence: (log excerpt, file paths)
- Suspected cause: (file:line if you can point to it; mark SPECULATION if unsure)
- Workaround used (if any)

## 10. Design flaws and mistakes
Problems that are not crashes: wrong assumptions, misleading metrics, bad defaults, confusing UX,
README errors. Same evidence rules.

## 11. Performance
Per-stage timings vs the README's RTX 4090 estimates; VRAM peaks; bottlenecks.

## 12. How I would test this properly
(see §10)

## 13. Suggestions, prioritised
Top 10 improvements, highest impact first, each with expected benefit and rough effort.

## 14. Not tested
What you skipped and why.

## Appendix
Commands run (in order), versions (`pip freeze` of key packages), list of attached files.
```

Severity guide:
- **critical:** crash, data loss, or unusable output on a core path.
- **high:** wrong result on a common case, or a core feature not working.
- **medium:** wrong result on an edge case, or a significant quality problem.
- **low:** cosmetic, docs, polish.

Finish by double-checking that every bug has reproduction steps and evidence, and that no secrets
appear anywhere in the report or the zip.
