"""SAM 3.1 on a Modal serverless GPU.

STATUS: written against the public SAM 3 API (Sam3Processor.set_image / set_text_prompt /
add_geometric_prompt) but NOT yet run against real weights.  The first run should be
`modal run infra/modal_sam.py::inspect --image path/to/sheet.png`, which prints the keys and shapes
the model returns so the adapter in `_extract` can be confirmed in one step.

Setup (once)
  pip install modal && modal setup
  # request access to the gated weights on Hugging Face first (facebook/sam3.1), then:
  modal secret create huggingface-secret HF_TOKEN=hf_xxx
  modal deploy infra/modal_sam.py          # creates app "pid-sam", class "SamServer"

Cost control
  * the container scales to zero after SCALEDOWN_S seconds idle
  * set a spend limit in the Modal dashboard (Settings -> Usage) before running evaluations
  * weights are cached in a Modal Volume, so they download only once

Environment overrides (set when deploying):  SAM_HF_REPO (default facebook/sam3.1), SAM_CKPT (path inside the volume)
"""
from __future__ import annotations

import io
import os

import modal

APP_NAME = "pid-sam"
SCALEDOWN_S = 120
GPU = os.getenv("SAM_GPU", "L4")
HF_REPO = os.getenv("SAM_HF_REPO", "facebook/sam3.1")

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("pid-sam-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("pillow", "numpy", "huggingface_hub")
    .run_commands("pip install git+https://github.com/facebookresearch/sam3.git")
)


def _to_list(x):
    try:
        return x.detach().float().cpu().numpy().tolist()
    except AttributeError:
        return list(x)


def _extract(state: dict, width: int, height: int) -> list[tuple[list[float], float]]:
    """Adapter from SAM's per-prompt state to [(xyxy pixel box, score)].

    Defensive about coordinate conventions: normalized boxes are scaled to pixels.
    Run `inspect` once to confirm the keys on the deployed version.
    """
    boxes = state.get("boxes")
    scores = state.get("scores")
    if boxes is None or scores is None:
        return []
    boxes, scores = _to_list(boxes), _to_list(scores)
    out = []
    for b, s in zip(boxes, scores):
        x0, y0, x1, y1 = [float(v) for v in b]
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:  # normalized
            x0, x1, y0, y1 = x0 * width, x1 * width, y0 * height, y1 * height
        out.append(([x0, y0, x1, y1], float(s)))
    return out


@app.cls(
    gpu=GPU,
    image=image,
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=SCALEDOWN_S,
    timeout=900,
)
class SamServer:
    @modal.enter()
    def load(self):
        import glob

        import torch
        from huggingface_hub import snapshot_download
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        ckpt = os.getenv("SAM_CKPT")
        if not ckpt:
            local = snapshot_download(HF_REPO, local_dir="/weights/sam3.1", token=os.environ.get("HF_TOKEN"))
            cands = sorted(glob.glob(os.path.join(local, "*.pt")) + glob.glob(os.path.join(local, "*.safetensors")))
            ckpt = cands[0] if cands else None
            weights.commit()
        try:
            self.model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False)
        except TypeError:
            # older builder signature: falls back to the package default checkpoint
            self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model, confidence_threshold=0.3)
        print(f"SAM loaded from {ckpt}")

    def _run(self, img, prompts, score_thr, refine):
        import torch

        w, h = img.size
        results = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(img)  # image features are computed once, reused per prompt
            for p in prompts:
                self.processor.reset_all_prompts(state)
                st = self.processor.set_text_prompt(state=state, prompt=p)
                found = [(b, s) for b, s in _extract(st, w, h) if s >= score_thr]
                results += [{"box": b, "score": s, "prompt": p} for b, s in found]
                if refine and found:
                    # self-exemplar refinement: use the best text hit as a positive box exemplar
                    (bx0, by0, bx1, by1), _ = max(found, key=lambda t: t[1])
                    cx, cy = (bx0 + bx1) / 2 / w, (by0 + by1) / 2 / h
                    bw, bh = (bx1 - bx0) / w, (by1 - by0) / h
                    self.processor.reset_all_prompts(state)
                    st2 = self.processor.add_geometric_prompt(state=state, box=[cx, cy, bw, bh], label=True)
                    results += [{"box": b, "score": s, "prompt": p} for b, s in _extract(st2, w, h) if s >= score_thr]
        return results

    @modal.method()
    def segment(self, png: bytes, prompts: list[str], score_thr: float = 0.35, refine: bool = True) -> list[dict]:
        from PIL import Image

        img = Image.open(io.BytesIO(png)).convert("RGB")
        return self._run(img, prompts, score_thr, refine)

    @modal.method()
    def inspect(self, png: bytes, prompt: str = "valve") -> dict:
        """Debug helper: what does the model actually return?"""
        from PIL import Image

        img = Image.open(io.BytesIO(png)).convert("RGB")
        state = self.processor.set_image(img)
        st = self.processor.set_text_prompt(state=state, prompt=prompt)
        info = {}
        for k, v in st.items():
            shape = getattr(v, "shape", None)
            info[k] = {"type": type(v).__name__, "shape": list(shape) if shape is not None else None}
        info["_sample_boxes"] = _to_list(st["boxes"])[:3] if "boxes" in st else None
        info["_sample_scores"] = _to_list(st["scores"])[:3] if "scores" in st else None
        return info


@app.local_entrypoint()
def inspect(image: str, prompt: str = "valve"):
    with open(image, "rb") as f:
        data = f.read()
    print(SamServer().inspect.remote(data, prompt))
