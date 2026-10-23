"""Pipeline orchestrator.

Each stage is a small agent with one job; they share a :class:`RunContext`
and record everything in ``<run_dir>/manifest.json`` so a run can be
inspected or resumed (``--resume``) without redoing the GPU stages.

  1 ConceptAgent         prompt/image -> AssetSpec + reference images
  2 ReconstructionAgent  reference image -> raw mesh
  3 TopologyAgent        Blender cleanup / quad remesh / metric scale
  4 SurfacingAgent       Blender UVs + PBR bake
  5 RiggingAgent         Blender segmentation (props) or armature (characters)
  6 AuditAgent           Blender renders + vision/metric audit -> fixes for 3/4
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .blender_automation import BlenderRunner
from .concept_planner import AssetSpec, ConceptPlanner, spec_from_knowledge_base
from .config import Config, find_blender
from .llm_client import LLMClient, LLMError
from .mesh_reconstruction import MeshReconstructor
from .reference_generator import ReferenceGenerator, ReferenceSet
from .shape_templates import DEFAULT_MATERIALS, ShapeComponent, expand_arrays, recipe_for
from .vision_auditor import AuditReport, VisionAuditor, apply_fixes, apply_rig_fixes

log = logging.getLogger(__name__)

ELONGATED = ("weapon", "vehicle")


@dataclass
class RunOptions:
    prompt: Optional[str] = None
    image_input: Optional[str] = None
    target_format: str = "glb"
    enable_rigging: bool = False
    offline: bool = False  # no LLM / VLM calls (knowledge base + metrics-only audit)
    skip_references: bool = False  # text input without diffusion (primitive backend testing)
    print_clearance_mm: float = 0.0
    rig_method: str = "rigify"
    resume: Optional[str] = None


@dataclass
class RunContext:
    run_dir: Path
    options: RunOptions
    spec: Optional[AssetSpec] = None
    references: Optional[ReferenceSet] = None
    raw_mesh: Optional[str] = None
    raw_camera: Optional[Dict[str, Any]] = None  # reconstruction model's input camera (file coords)
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    export: Optional[Dict[str, Any]] = None
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def save(self, cfg: Config):
        data = {
            "options": dataclasses.asdict(self.options),
            "config": cfg.to_dict(),
            "spec": self.spec.to_dict() if self.spec else None,
            "references": self.references.to_dict() if self.references else None,
            "raw_mesh": self.raw_mesh,
            "raw_camera": self.raw_camera,
            "iterations": self.iterations,
            "export": self.export,
            "timings": self.timings,
        }
        self.manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _slug_dir(base: Path, name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path, n = base / f"{name}_{stamp}", 1
    while path.exists():
        n += 1
        path = base / f"{name}_{stamp}_{n}"
    path.mkdir(parents=True)
    return path


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm: Optional[LLMClient] = None
        self.vlm: Optional[LLMClient] = None
        self._blender: Optional[BlenderRunner] = None

    # ----------------------------------------------------------------- setup

    def _init_clients(self, offline: bool):
        if offline:
            log.info("Offline mode: knowledge-base planning and metrics-only audit")
            return
        self.llm = LLMClient.from_config(self.cfg.llm)
        self.vlm = LLMClient.from_config(self.cfg.vision)
        if self.cfg.pipeline.unload_models_between_stages:
            # Ollama otherwise keeps both models resident (~15 GB) while SDXL and TripoSR load
            self.llm.keep_alive = self.vlm.keep_alive = "0"
        if not self.llm.ping():
            log.warning("Planner LLM %s at %s is unreachable; using knowledge base",
                        self.cfg.llm.model, self.cfg.llm.base_url)
            self.llm = None
        if not self.vlm.ping():
            log.warning("Vision LLM %s at %s is unreachable; audit will use metrics only",
                        self.cfg.vision.model, self.cfg.vision.base_url)
            self.vlm = None

    def blender(self, run_dir: Path) -> BlenderRunner:
        if self._blender is None:
            exe = find_blender(self.cfg.blender.executable)
            self._blender = BlenderRunner(exe or "", run_dir / "blender_jobs", self.cfg.blender.timeout_s,
                                          self.cfg.blender.use_gpu)
            log.info("Using %s", self._blender.version())
        return self._blender

    @staticmethod
    def _timed(ctx: RunContext, name: str, start: float):
        ctx.timings[name] = round(ctx.timings.get(name, 0.0) + time.time() - start, 2)

    # ------------------------------------------------------------------- run

    def run(self, options: RunOptions) -> RunContext:
        if not options.prompt and not options.image_input and not options.resume:
            raise ValueError("provide --prompt, --image_input or --resume")
        self._init_clients(options.offline)
        if options.resume:
            ctx = self._load(Path(options.resume), options)
        else:
            base = Path(self.cfg.pipeline.output_dir)
            base.mkdir(parents=True, exist_ok=True)
            seed_name = spec_from_knowledge_base(options.prompt or Path(options.image_input).stem).name
            ctx = RunContext(_slug_dir(base, seed_name), options)
        log.info("Run directory: %s", ctx.run_dir)

        if ctx.spec is None or ctx.references is None and not options.skip_references:
            self.stage1_concept(ctx)
            ctx.save(self.cfg)
        if ctx.raw_mesh is None or not Path(ctx.raw_mesh).is_file():
            self.stage2_reconstruct(ctx)
            ctx.save(self.cfg)
        best = self.refinement_loop(ctx)
        self.export(ctx, best)
        ctx.save(self.cfg)
        return ctx

    def _load(self, run_dir: Path, options: RunOptions) -> RunContext:
        data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        saved = RunOptions(**data["options"])
        # CLI flags for the downstream stages win over the saved ones
        saved.target_format = options.target_format or saved.target_format
        saved.enable_rigging = options.enable_rigging or saved.enable_rigging
        saved.offline = options.offline
        saved.resume = str(run_dir)
        ctx = RunContext(run_dir, saved)
        ctx.spec = AssetSpec.from_dict(data["spec"]) if data.get("spec") else None
        if data.get("references"):
            ctx.references = ReferenceSet(**data["references"])
        ctx.raw_mesh = data.get("raw_mesh")
        ctx.raw_camera = data.get("raw_camera")
        if ctx.raw_camera is None and ctx.raw_mesh and Path(ctx.raw_mesh).name == "raw_triposr.glb":
            from .mesh_reconstruction import TripoSRBackend

            ctx.raw_camera = TripoSRBackend.input_camera()  # runs from before cameras were recorded
        ctx.timings = data.get("timings", {})
        log.info("Resuming %s (spec=%s, raw_mesh=%s)", run_dir, bool(ctx.spec), ctx.raw_mesh)
        return ctx

    # --------------------------------------------------------------- stage 1

    def describe_image(self, image: str) -> str:
        if self.vlm is None:
            return Path(image).stem.replace("_", " ")
        try:
            reply = self.vlm.chat(
                "Name the single main object in this image as a short 3D asset brief (max 12 words), "
                "e.g. 'a weathered wooden treasure chest with iron bands'. Reply with the brief only.",
                images=[image])
            return reply.strip().strip('"').splitlines()[0][:120] or Path(image).stem
        except LLMError as exc:
            log.warning("Image description failed: %s", exc)
            return Path(image).stem.replace("_", " ")

    def stage1_concept(self, ctx: RunContext):
        start = time.time()
        opts = ctx.options
        prompt = opts.prompt or self.describe_image(opts.image_input)
        log.info("STAGE 1 - concept: %r", prompt)
        if ctx.spec is None:
            ctx.spec = ConceptPlanner(self.llm).plan(prompt)
        spec = ctx.spec
        log.info("Spec: %s [%s] %.3f x %.3f x %.3f m, %d parts, rig=%s", spec.name, spec.category,
                 spec.length_m, spec.width_m, spec.height_m, len(spec.parts), spec.rig_type)
        (ctx.run_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

        ref_dir = ctx.run_dir / "references"
        if opts.image_input:
            gen = ReferenceGenerator(self.cfg.diffusion, self.cfg.preprocess, self.cfg.reconstruction.device,
                                     self.cfg.pipeline.seed)
            ctx.references = gen.from_image(opts.image_input, ref_dir)
        elif not opts.skip_references and self.cfg.reconstruction.backend != "primitive":
            # (the placeholder backend needs no images, so diffusion is skipped for it)
            gen = ReferenceGenerator(self.cfg.diffusion, self.cfg.preprocess, self.cfg.reconstruction.device,
                                     self.cfg.pipeline.seed)
            ctx.references = gen.from_text(spec, ref_dir)
        self._timed(ctx, "stage1_concept", start)

    # --------------------------------------------------------------- stage 2

    def stage2_reconstruct(self, ctx: RunContext):
        start = time.time()
        log.info("STAGE 2 - image-to-3D (%s)", self.cfg.reconstruction.backend)
        refs = ctx.references
        if refs is None and self.cfg.reconstruction.backend != "primitive":
            raise RuntimeError("no reference image available for reconstruction")
        primary_view = None
        if refs is not None:
            primary_view = next((v for v, p in refs.views.items() if p["prepared"] == refs.primary), None)
        rgba = refs.views[primary_view]["rgba"] if refs and primary_view else (refs.primary if refs else "")
        recon = MeshReconstructor(self.cfg.reconstruction, ctx.spec.dimensions, self.cfg.pipeline.seed)
        try:
            raw = recon.reconstruct(refs.primary if refs else "", rgba, ctx.run_dir / "raw")
        finally:
            if self.cfg.pipeline.unload_models_between_stages:
                recon.unload()
        ctx.raw_mesh = raw.path
        ctx.raw_camera = raw.camera
        self._timed(ctx, "stage2_reconstruct", start)

    # ----------------------------------------------------------- stages 3-6

    def _cleanup_params(self, spec: AssetSpec, camera: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dataclasses.asdict(self.cfg.cleanup)
        params["target_faces"] = params["target_faces"] or spec.face_budget or 12000
        if spec.is_organic:
            params["max_tilt_correction"] = 0.0  # yaw only: an A-pose body has no upright min box
        params.update(name=spec.name, dimensions=spec.dimensions, fit_dimension=spec.fit_dimension,
                      axis_scale={}, reference_camera=camera, organic=spec.is_organic,
                      # Bodies keep their image proportions; objects are fitted to real-world size
                      per_axis_fit=not spec.is_organic,
                      symmetrize=spec.is_organic and params.pop("symmetrize_characters"))
        params.pop("symmetrize_characters", None)
        return params

    @staticmethod
    def _mirror_axes(spec: AssetSpec) -> List[str]:
        """Symmetry planes safe for filling unseen faces from the reference image."""
        if spec.is_organic:
            return ["X"]  # left/right only - never paint the face onto the back of the head
        if spec.category in ELONGATED:
            return ["Y"]  # left/right flank - never swap muzzle and stock
        return ["X", "Y"]

    def _shape_components(self, spec: AssetSpec) -> Optional[List[Dict[str, Any]]]:
        """Template recipe for Stage 3b: planner shapes > library recipe > [] (auto-classify).

        None disables the stage (organic assets or shapes.mode == "mesh").
        """
        if self.cfg.shapes.mode == "mesh" or spec.is_organic:
            return None
        # Curated library recipes beat planner-improvised ones (a car must keep its wheels)
        comps = [c.to_dict() for c in recipe_for(spec.prompt, spec.category)] or spec.shapes
        if not comps and spec.category in ELONGATED and not self.cfg.shapes.blueprint:
            return None
        return [c.to_dict() for c in expand_arrays([ShapeComponent(**c) for c in comps])]

    @staticmethod
    def _materials(spec: AssetSpec, components: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Spec palette plus PBR defaults for any material a template references."""
        mats = [dataclasses.asdict(m) for m in spec.materials]
        names = {m["name"] for m in mats}
        for comp in components or []:
            name = comp.get("material")
            if name and name not in names and name in DEFAULT_MATERIALS:
                mats.append(dict(name=name, coverage=0.1, **DEFAULT_MATERIALS[name]))
                names.add(name)
        return mats

    def _shape_stage(self, runner, ctx: RunContext, blend: str, it_dir: Path, it: int,
                     components: List[Dict[str, Any]]) -> Dict[str, Any]:
        spec = ctx.spec
        shapes = self.cfg.shapes
        out = str(it_dir / "stage3b_templates.blend")
        materials = self._materials(spec, components)
        try:
            result = runner.run("primitives", dict(
                input_blend=blend, output_blend=out, components=components, mode=shapes.mode,
                recipe_max_error=shapes.recipe_max_error, classify_max_error=shapes.classify_max_error,
                union=shapes.union, refine=shapes.refine, materials=materials,
                # A single box/cylinder guess would erase a weapon's or vehicle's detail
                classify=spec.category not in ELONGATED, blueprint=shapes.blueprint,
                blueprint_max_error=shapes.blueprint_max_error, blueprint_resolution=shapes.blueprint_resolution,
                parts=[dataclasses.asdict(p) for p in spec.parts],
                default_material=max(spec.materials, key=lambda m: m.coverage).name if spec.materials else "",
                auto_smooth_angle=self.cfg.cleanup.auto_smooth_angle, seed=self.cfg.pipeline.seed,
            ), tag=f"it{it}_templates")
        except Exception as exc:  # templates are an enhancement; never lose the run over them
            log.warning("Primitive templates failed (%s); keeping the scan mesh", exc)
            return {"used": False, "error": str(exc)}
        if result.get("used"):
            result["blend"] = out
            log.info("Rebuilt from primitive templates (%s, fit error %.3f, parts %s)",
                     result["source"], result["fit_error"], result.get("parts_built") or "-")
        else:
            log.info("Templates rejected (%s fit error %.3f); keeping the cleaned scan",
                     result.get("source"), result.get("fit_error", -1))
        return result

    def _texture_params(self, spec: AssetSpec, refs: Optional[ReferenceSet],
                        camera: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dataclasses.asdict(self.cfg.texture)
        primary = None
        if refs is not None:
            primary = next((v for v in refs.views.values() if v.get("prepared") == refs.primary), None)
        params["reference_image"] = (primary or {}).get("projection")
        params["reference_camera"] = camera
        params["mirror_axes"] = self._mirror_axes(spec) if params.pop("mirror_fill") else []
        params["materials"] = self._materials(spec, self._shape_components(spec))
        projections = []
        if refs and params["albedo_source"] == "projection":
            # Diffusion "side" views of long objects show their profile, i.e. Blender's XZ plane
            xz, yz = ("side", "front") if spec.category in ELONGATED else ("front", "side")
            for view, plane in ((xz, "XZ"), (yz, "YZ")):
                if view in refs.views:
                    projections.append({"image": refs.views[view]["rgba"], "plane": plane, "flip_u": False})
            if not projections:
                log.warning("Projection texturing needs front/side views; falling back to bake")
                params["albedo_source"] = "bake"
        params["projections"] = projections
        return params

    def refinement_loop(self, ctx: RunContext) -> Dict[str, Any]:
        spec = ctx.spec
        runner = self.blender(ctx.run_dir)
        auditor = VisionAuditor(self.vlm, self.cfg.pipeline.approval_score)
        cleanup = self._cleanup_params(spec, ctx.raw_camera)
        texture = self._texture_params(spec, ctx.references, ctx.raw_camera)
        reference_image = ctx.references.concept_sheet if ctx.references else None
        stage5_mode = self._stage5_mode(ctx)
        located_parts = None
        prev_cleanup, prev_geometry, shape_result = None, None, {}
        components = self._shape_components(spec)
        rig = {"method": ctx.options.rig_method, "weights": "auto", "weight_smooth": 1, "max_influences": 4,
               "pose_test": True}
        ctx.iterations = []

        for it in range(1, max(1, self.cfg.pipeline.max_iterations) + 1):
            it_dir = ctx.run_dir / f"iter_{it}"
            it_dir.mkdir(exist_ok=True)
            record: Dict[str, Any] = {"iteration": it, "cleanup_params": cleanup, "texture_params": texture}
            log.info("ITERATION %d", it)

            # STAGE 3 (+ the Stage 5 segmentation cut), skipped when only texture params changed
            if cleanup == prev_cleanup and prev_geometry:
                geometry_blend = prev_geometry
                record["stage3"] = {"reused": prev_geometry}
            else:
                start = time.time()
                log.info("STAGE 3 - topology cleanup / remesh / scale (%s)", cleanup["remesh_mode"])
                geometry_blend = str(it_dir / "stage3_topology.blend")
                record["stage3"] = runner.run("cleanup", dict(cleanup, input_mesh=ctx.raw_mesh,
                                                              output_blend=geometry_blend), tag=f"it{it}_cleanup")
                self._timed(ctx, "stage3_topology", start)

                # STAGE 3b - rebuild hard-surface assets from primitive templates
                shape_result = {}
                if components is not None:
                    start = time.time()
                    log.info("STAGE 3b - primitive templates (%s)",
                             f"{len(components)} components" if components else "auto-classify")
                    shape_result = self._shape_stage(runner, ctx, geometry_blend, it_dir, it, components)
                    record["stage3b"] = shape_result
                    if shape_result.get("used"):
                        geometry_blend = shape_result["blend"]
                    self._timed(ctx, "stage3b_templates", start)

                built = set(shape_result.get("parts_built") or []) if shape_result.get("used") else set()
                if stage5_mode == "segment" and any(p.name not in built for p in spec.parts):
                    # Parts are cut BEFORE surfacing so their cut faces get real UVs and
                    # body + parts share one texture set; pivots/limits are part of Stage 5.
                    start = time.time()
                    if located_parts is None:
                        located_parts = self._locate_parts(runner, auditor, geometry_blend, it_dir, spec, it)
                    todo = [p for p in located_parts if p["name"] not in built]
                    log.info("STAGE 5 - segmentation of %d parts", len(todo))
                    segmented = str(it_dir / "stage5_segmented.blend")
                    record["stage5"] = runner.run("segment", dict(
                        input_blend=geometry_blend, output_blend=segmented, parts=todo,
                        clearance_mm=ctx.options.print_clearance_mm), tag=f"it{it}_segment")
                    geometry_blend = segmented
                    self._timed(ctx, "stage5_rigging", start)
            prev_cleanup, prev_geometry = cleanup, geometry_blend

            # STAGE 4
            start = time.time()
            if shape_result.get("used"):
                # Primitives sit up to a few cm from the lumpy scan: widen the bake cage to reach it,
                # and skip the scan's normals so the clean shapes shade clean
                texture = dict(texture, bake_normal=self.cfg.shapes.bake_normal,
                               cage_extrusion=max(texture["cage_extrusion"], shape_result.get("cage_hint", 0.0)))
            log.info("STAGE 4 - UVs + PBR (%s, %dpx)", texture["uv_method"], texture["resolution"])
            s4_blend = str(it_dir / "stage4_surfacing.blend")
            record["stage4"] = runner.run("texture", dict(
                texture, input_blend=geometry_blend, output_blend=s4_blend, texture_dir=str(it_dir / "textures"),
                use_gpu=self.cfg.blender.use_gpu), tag=f"it{it}_texture")
            final_blend = s4_blend
            self._timed(ctx, "stage4_surfacing", start)

            # STAGE 5 (characters/creatures): skinning does not touch UVs, so it follows surfacing
            if stage5_mode == "rig":
                start = time.time()
                log.info("STAGE 5 - %s rig (%s)", spec.rig_type, ctx.options.rig_method)
                final_blend = str(it_dir / "stage5_rigged.blend")
                record["stage5"] = runner.run("rig", dict(
                    rig, input_blend=s4_blend, output_blend=final_blend, rig_type=spec.rig_type),
                    tag=f"it{it}_rig")
                record["rig_params"] = rig
                self._timed(ctx, "stage5_rigging", start)

            # STAGE 6
            start = time.time()
            log.info("STAGE 6 - audit renders + review")
            render = runner.run("render", dict(
                input_blend=final_blend, output_dir=str(ctx.run_dir / "renders" / f"iter_{it}"),
                resolution=self.cfg.render.resolution, engine=self.cfg.render.engine,
                samples=self.cfg.render.samples, include_uv_checker=self.cfg.render.include_uv_checker,
                use_gpu=self.cfg.blender.use_gpu), tag=f"it{it}_render")
            assembled = [round(b - a, 4) for a, b in zip(render["bbox_min"], render["bbox_max"])]
            renders = dict(render["renders"])
            rig_report = record.get("stage5") if stage5_mode == "rig" else None
            if rig_report and rig_report.get("posed_blend"):
                posed = runner.run("render", dict(
                    input_blend=rig_report["posed_blend"],
                    output_dir=str(ctx.run_dir / "renders" / f"iter_{it}" / "posed"),
                    resolution=self.cfg.render.resolution, engine=self.cfg.render.engine,
                    samples=self.cfg.render.samples, views=["front", "shaded"], use_gpu=self.cfg.blender.use_gpu),
                    tag=f"it{it}_render_posed")
                renders.update({f"posed_{k}": v for k, v in posed["renders"].items()})
                render["renders"] = renders
            report: AuditReport = auditor.audit(it, renders, render["metrics"], spec,
                                                cleanup["target_faces"], reference_image, assembled, rig_report,
                                                record.get("stage4"))
            record.update(render=render, audit=report.to_dict(), final_blend=final_blend)
            self._timed(ctx, "stage6_audit", start)
            ctx.iterations.append(record)
            ctx.save(self.cfg)

            if report.approved:
                log.info("Approved on iteration %d", it)
                break
            if it == self.cfg.pipeline.max_iterations:
                log.warning("Not approved after %d iterations; exporting best attempt", it)
                break
            cleanup, texture, changes = apply_fixes(cleanup, texture, report.issues)
            rig, rig_changes = apply_rig_fixes(rig, report.issues)
            changes += rig_changes
            record["fixes"] = changes
            if not changes:
                log.warning("Audit found no actionable fixes; stopping refinement")
                break
            log.info("Feeding back fixes: %s", "; ".join(changes))

        best = max(ctx.iterations, key=lambda r: (r["audit"]["approved"], r["audit"]["score"], r["iteration"]))
        log.info("Selected iteration %d (score %.2f, approved=%s)", best["iteration"],
                 best["audit"]["score"], best["audit"]["approved"])
        return best

    def _stage5_mode(self, ctx: RunContext) -> Optional[str]:
        spec = ctx.spec
        if not ctx.options.enable_rigging:
            return None
        if spec.is_organic:
            if spec.rig_type == "none":
                spec.rig_type = "quadruped" if spec.category == "creature" else "biped"
            return "rig"
        if spec.parts:
            return "segment"
        log.info("Rigging enabled but %s has no movable parts; skipping Stage 5", spec.name)
        return None

    def _locate_parts(self, runner, auditor, blend, it_dir, spec, it):
        parts = [dataclasses.asdict(p) for p in spec.parts]
        if auditor.llm is None:
            return parts
        pre = runner.run("render", dict(input_blend=blend, output_dir=str(it_dir / "part_locator"),
                                        resolution=self.cfg.render.resolution, engine=self.cfg.render.engine,
                                        samples=max(8, self.cfg.render.samples // 2), views=["front"],
                                        use_gpu=self.cfg.blender.use_gpu), tag=f"it{it}_locate")
        return auditor.locate_parts(pre["renders"]["front"], pre["framing"]["front"],
                                    pre["bbox_min"], pre["bbox_max"], spec)

    # ---------------------------------------------------------------- export

    def export(self, ctx: RunContext, best: Dict[str, Any]):
        fmt = ctx.options.target_format.lower()
        export_dir = ctx.run_dir / "export"
        runner = self.blender(ctx.run_dir)
        result = runner.run("export", dict(
            input_blend=best["final_blend"], output_path=str(export_dir / f"{ctx.spec.name}.{fmt}"),
            output_blend=str(export_dir / f"{ctx.spec.name}.blend"), format=fmt), tag="export")
        result["iteration"] = best["iteration"]
        result["approved"] = best["audit"]["approved"]
        result["score"] = best["audit"]["score"]
        renders = best["render"]["renders"]
        preview_dir = export_dir / "previews"
        preview_dir.mkdir(exist_ok=True)
        for view, path in renders.items():
            shutil.copy2(path, preview_dir / f"{view}.png")
        ctx.export = result
        log.info("Exported %s (%.1f MB)", result["output"], result["bytes"] / 1e6)
