from asset_pipeline.concept_planner import AssetSpec, ConceptPlanner, lookup_knowledge_base, spec_from_knowledge_base
from asset_pipeline.llm_client import LLMError

from conftest import FakeLLM


def test_knowledge_base_longest_keyword_wins():
    assert lookup_knowledge_base("a castle gate with portcullis")["dims"][0] == 5.0
    assert lookup_knowledge_base("Medieval Castle Wall")["category"] == "architecture"
    assert lookup_knowledge_base("A functional MP5 submachine gun")["dims"] == (0.68, 0.06, 0.26)
    assert lookup_knowledge_base("a spaceship hull") is None


def test_offline_spec_for_mp5():
    spec = ConceptPlanner(None).plan("A functional MP5 submachine gun")
    assert spec.category == "weapon"
    assert spec.length_m == 0.68 and spec.fit_dimension == "length"
    assert {p.name for p in spec.parts} == {"magazine", "trigger", "cocking_handle"}
    assert set(spec.view_prompts) == {"isometric", "front", "side"}


def test_llm_reply_is_validated_and_clamped():
    reply = {
        "name": "MP5 SMG!",
        "category": "weapon",
        "length_m": 0.7, "width_m": 0.07, "height_m": 0.27,
        "fit_dimension": "length",
        "materials": [{"name": "steel", "base_color": [2, -1, 0.5], "metallic": 3, "roughness": 0.4}],
        "parts": [{"name": "Magazine", "region_min": [0.6, 0, 0.5], "region_max": [0.5, 1, 0.0],
                   "pivot": [0.5], "motion": "slide", "axis": "q"}],
        "rig_type": "dragon",
        "face_budget": "lots",
    }
    spec = ConceptPlanner(FakeLLM([reply])).plan("a sci-fi gadget")  # unknown object: LLM values are used
    assert spec.name == "mp5_smg"
    assert spec.materials[0].base_color == [1.0, 0.0, 0.5]
    assert spec.materials[0].metallic == 1.0
    part = spec.parts[0]
    assert part.name == "magazine"
    assert part.region_min == [0.5, 0.0, 0.0] and part.region_max == [0.6, 1.0, 0.5]  # min/max swapped back
    assert part.pivot == [0.5, 0.5, 0.5]
    assert part.motion == "fixed" and part.axis == "Y"
    assert spec.rig_type == "none"
    assert spec.face_budget == 12000


def test_llm_scale_outlier_falls_back_to_reference():
    reply = {"name": "door", "category": "architecture", "length_m": 90, "width_m": 5, "height_m": 210}
    spec = ConceptPlanner(FakeLLM([reply])).plan("a wooden door")
    assert spec.height_m == 2.1  # LLM answered in centimetres; reference dims win


def test_llm_failure_falls_back():
    spec = ConceptPlanner(FakeLLM([LLMError("down")])).plan("a wooden chair")
    assert spec.category == "furniture" and spec.height_m == 0.9


def test_characters_get_rig_not_parts():
    reply = {"name": "knight", "category": "character", "length_m": 0.6, "width_m": 0.3, "height_m": 1.85,
             "parts": [{"name": "sword"}]}
    spec = ConceptPlanner(FakeLLM([reply])).plan("a knight")
    assert spec.parts == [] and spec.rig_type == "biped"


def test_spec_round_trip():
    spec = spec_from_knowledge_base("a sedan car")
    again = AssetSpec.from_dict(spec.to_dict())
    assert again == spec


def test_reference_wins_for_known_objects():
    # Sword "3 cm tall" (axis-convention mix-up) and a 0.3 m chest are replaced by reference dims
    sword = ConceptPlanner(FakeLLM([{"name": "s", "category": "weapon", "length_m": 1.2, "width_m": 0.15,
                                     "height_m": 0.03}])).plan("a longsword")
    assert sword.dimensions == [1.0, 0.05, 0.2]
    chest = ConceptPlanner(FakeLLM([{"name": "c", "category": "furniture", "length_m": 0.6, "width_m": 0.45,
                                     "height_m": 0.3}])).plan("a wooden treasure chest")
    assert chest.dimensions == [0.9, 0.55, 0.6] and chest.category == "prop"
    # Close enough: the planner's numbers are kept
    near = ConceptPlanner(FakeLLM([{"name": "m", "category": "weapon", "length_m": 0.7, "width_m": 0.07,
                                    "height_m": 0.27}])).plan("an MP5 smg")
    assert near.dimensions == [0.7, 0.07, 0.27]
    # Improvised parts lose to the reference parts
    mp5 = ConceptPlanner(FakeLLM([{"name": "m", "category": "weapon", "length_m": 0.68, "width_m": 0.06,
                                   "height_m": 0.26, "parts": [{"name": "buttstock"}]}])).plan("MP5 submachine gun")
    assert {p.name for p in mp5.parts} == {"magazine", "trigger", "cocking_handle"}
    assert "dimensions_m" in mp5.to_dict()
