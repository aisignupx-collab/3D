from types import SimpleNamespace

from PIL import Image, ImageDraw

from asset_pipeline.config import DiffusionConfig, PreprocessConfig
from asset_pipeline.reference_generator import ComfyUIBackend, ImagePreprocessor, build_concept_sheet


def cutout(w=300, h=200, box=(50, 80, 250, 120)):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle(box, fill=(200, 30, 30, 255))
    return img


def test_tight_crop():
    crop = ImagePreprocessor.tight_crop(cutout())
    assert crop.size == (201, 41)


def test_prepare_centres_at_foreground_ratio():
    pre = ImagePreprocessor(PreprocessConfig(output_size=512, foreground_ratio=0.8))
    out = pre.prepare(cutout())
    assert out.size == (512, 512) and out.mode == "RGB"
    assert out.getpixel((0, 0)) == (127, 127, 127)
    # foreground spans 80% of the canvas along its longest side, centred
    row = [out.getpixel((x, 256)) for x in range(512)]
    red = [x for x, px in enumerate(row) if px[0] > 150 and px[1] < 100]
    assert abs((red[-1] - red[0] + 1) - 410) <= 2
    assert abs((red[0] + red[-1]) / 2 - 255.5) <= 2


def test_isolate_skips_rembg_for_transparent_input():
    img = cutout()
    assert ImagePreprocessor(PreprocessConfig()).isolate(img) is img


def test_process_and_concept_sheet(tmp_path):
    pre = ImagePreprocessor(PreprocessConfig(output_size=256))
    views = {"front": pre.process(cutout(), "front", tmp_path), "side": pre.process(cutout(), "side", tmp_path)}
    for paths in views.values():
        for path in paths.values():
            assert Image.open(path).size
    sheet = Image.open(build_concept_sheet(views, tmp_path / "sheet.png", "test"))
    assert sheet.size == (1024, 552)


def test_comfyui_placeholders_keep_types():
    backend = ComfyUIBackend(DiffusionConfig())
    graph = backend._fill(backend._default_workflow(), {"prompt": "a crate", "negative": "blur", "seed": 7,
                                                        "width": 1024, "height": 768})
    assert graph["3"]["inputs"]["seed"] == 7
    assert graph["5"]["inputs"]["height"] == 768
    assert graph["6"]["inputs"]["text"] == "a crate"
    assert backend._fill("seed={{seed}}", {"seed": 7}) == "seed=7"
