import json

from conftest import ScriptedClient, text_reply
from PIL import Image

from pid.segment.muse_localizer import MuseLocalizer


def _reply(symbols):
    return text_reply(json.dumps({"symbols": symbols}))


def test_boxes_are_mapped_back_to_sheet_coordinates():
    img = Image.new("RGB", (1000, 600), "white")
    # tile 800 / overlap 0 -> two tiles side by side; model reports a box in each tile's own pixels
    client = ScriptedClient([_reply([{"class": "pump", "box": [100, 100, 160, 160]}]), _reply([{"class": "tank", "box": [50, 200, 130, 300]}])])
    dets = MuseLocalizer(client, tile=800, overlap=0).detect(img)
    boxes = sorted(d.bbox for d in dets)
    assert [100, 100, 160, 160] in [[round(v) for v in b] for b in boxes]
    assert [850, 200, 930, 300] in [[round(v) for v in b] for b in boxes]  # second tile starts at x=800


def test_duplicate_boxes_from_overlapping_tiles_are_merged():
    img = Image.new("RGB", (1000, 400), "white")
    box = {"class": "valve", "box": [10, 10, 60, 60]}
    # tile 700, overlap 400 -> second tile starts at x=300; same physical symbol reported at x=300+... 
    client = ScriptedClient([_reply([{"class": "valve", "box": [310, 10, 360, 60]}]), _reply([box])])
    dets = MuseLocalizer(client, tile=700, overlap=400).detect(img)
    assert len(dets) == 1


def test_bad_replies_are_skipped_not_fatal():
    img = Image.new("RGB", (500, 500), "white")
    client = ScriptedClient([text_reply("not json at all")])
    assert MuseLocalizer(client, tile=800, overlap=0).detect(img) == []
    client = ScriptedClient([_reply([{"class": "pump"}, {"class": "pump", "box": ["a", 1, 2, 3]}, {"class": "pump", "box": [50, 50, 40, 40]}])])
    assert MuseLocalizer(client, tile=800, overlap=0).detect(img) == []


def test_normalized_coordinates_are_scaled():
    img = Image.new("RGB", (400, 200), "white")
    client = ScriptedClient([_reply([{"class": "tank", "box": [0.25, 0.25, 0.5, 0.75]}])])
    dets = MuseLocalizer(client, tile=800, overlap=0).detect(img)
    assert [round(v) for v in dets[0].bbox] == [100, 50, 200, 150]


def test_prompt_lists_classes_and_sends_image():
    client = ScriptedClient([_reply([])])
    MuseLocalizer(client, tile=800, overlap=0).detect(Image.new("RGB", (300, 300), "white"))
    content = client.calls[0]["messages"][0]["content"]
    assert "check_valve" in content[0]["text"] and content[1]["type"] == "image_url"
    assert client.calls[0]["purpose"] == "localize"
