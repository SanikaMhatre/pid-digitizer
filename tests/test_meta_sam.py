import json

import httpx
from conftest import ScriptedClient  # noqa: F401
from PIL import Image

from pid.config import build_localizer
from pid.segment.meta_sam import MetaSamLocalizer, parse_boxes, response_text

TEXT = (
    "<0f>0<|box;x1=10;y1=20;x2=59;y2=79;w=400;h=300;c=0.9|><|mask;x=0;y=0;data=1,2,3|>,"
    "1<|box;x1=200;y1=100;x2=249;y2=149;w=400;h=300;c=0.1|><|mask;x=0;y=0;data=4|>"
)


def _handler(text_by_phrase):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((str(request.url), body))
        phrase = body["input"][0]["content"][0]["text"]
        return httpx.Response(200, json={"output": [{"content": [{"type": "output_text", "text": text_by_phrase.get(phrase, "")}]}]})

    return handler, seen


def test_parse_boxes_and_response_text():
    boxes = parse_boxes(TEXT)
    assert len(boxes) == 2 and boxes[0]["x1"] == 10 and boxes[0]["c"] == 0.9
    assert parse_boxes("<|box;x1=1;y1=a|>") == []
    assert response_text({"output_text": "x"}) == "x"
    assert response_text({"output": [{"content": [{"text": "a"}, {"text": "b"}]}]}) == "ab"


def test_request_shape_boxes_mapped_and_low_confidence_dropped(make_client):
    handler, seen = _handler({"pump": TEXT})
    client = make_client(handler, cache=False)
    loc = MetaSamLocalizer(client, prompts={"pump": "pump", "valve": "valve"})
    dets = loc.detect(Image.new("RGB", (400, 300), "white"))
    url, body = seen[0]
    assert url.endswith("/v1/responses") and body["model"] == "sam-3.1"
    part_types = [p["type"] for p in body["input"][0]["content"]]
    assert part_types == ["input_text", "input_image"]
    assert body["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert len(dets) == 1  # c=0.1 dropped; "valve" returned nothing
    assert [round(v) for v in dets[0].bbox] == [10, 20, 60, 80]  # x2 / y2 are inclusive
    assert dets[0].hint == "pump"


def test_tiles_offset_and_scale(make_client):
    handler, _ = _handler({"pump": "<0f>0<|box;x1=10;y1=10;x2=49;y2=49;w=100;h=100;c=0.9|>"})
    client = make_client(handler, cache=False)
    loc = MetaSamLocalizer(client, prompts={"pump": "pump"}, tile=100, overlap=0)
    dets = loc.detect(Image.new("RGB", (200, 100), "white"))
    assert sorted(round(d.bbox[0]) for d in dets) == [10, 110]  # the second tile is offset by its x origin


def test_cost_is_recorded_per_image(make_client):
    handler, _ = _handler({})
    client = make_client(handler, cache=False)
    MetaSamLocalizer(client, prompts={"pump": "pump"}).detect(Image.new("RGB", (300, 300), "white"))
    assert abs(client.ledger.totals()["cost_usd"] - 0.0025) < 1e-9


def test_build_localizer_choices(monkeypatch):
    from pid.models.llm import ChatClient
    from pid.segment.muse_localizer import MuseLocalizer

    c = ChatClient(api_key="k", cache_dir=False)
    assert isinstance(build_localizer(c, "sam"), MetaSamLocalizer)
    assert isinstance(build_localizer(c, "muse"), MuseLocalizer)
    monkeypatch.setenv("PID_LOCALIZER", "muse")
    assert isinstance(build_localizer(c), MuseLocalizer)
    try:
        build_localizer(c, "other")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
