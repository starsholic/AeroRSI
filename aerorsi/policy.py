"""The navigation policy: AeroVLA-2B (Qwen3-VL) with a LoRA adapter, as in AeroVLA/AeroDPO.

Input: a 384x768 mosaic of the front (top) and down (bottom) RGB views and the prompt
``"Fly <direction> and find the target. <object description>\\nAction: "``, where the direction is the target's
coarse bearing in the body frame. Output: a native action text (see ``aerorsi.actions``).

Every policy in the loop shares one trainable layout: rank-64 LoRA on the seven language projections plus a full copy
of the visual merger, loaded on top of the frozen SFT model. Round 0 (the SFT policy) is this layout with every LoRA B
at zero (``init_adapter``); each round continues training the adapter of the round before.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .actions import Prediction, decode
from .utils import file_hash

IMAGE_SIZE = 384
LORA = {"r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": ["visual.merger"], "bias": "none", "task_type": "CAUSAL_LM"}


def mosaic(observation) -> Image.Image:
    """Front view above the down view, each resized to 384x384 (observations hold BGR frames)."""
    result = Image.new("RGB", (IMAGE_SIZE, 2 * IMAGE_SIZE))
    for offset, camera in enumerate((0, 4)):
        frame = Image.fromarray(np.ascontiguousarray(observation["rgb"][camera][..., ::-1]))
        result.paste(frame.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC), (0, offset * IMAGE_SIZE))
    return result


def direction(state, target) -> str:
    """Coarse bearing of the target in the body frame, with a trailing space (empty when directly above)."""
    q = state["orientation"]
    q = [q.get("x", 0), q.get("y", 0), q.get("z", 0), q.get("w", 1)] if isinstance(q, dict) else q
    world = np.asarray(target) - np.asarray(state["position"])
    if np.linalg.norm(world[:2]) < 0.01:
        return ""
    body = Rotation.from_quat(q).inv().apply(world)
    angle = np.degrees(np.arctan2(body[1], body[0]))
    if -15 <= angle <= 15:
        return "straight ahead "
    if 15 < angle <= 60:
        return "forward-right "
    if 60 < angle <= 120:
        return "to your right "
    if 120 < angle <= 180:
        return "to your right rear "
    if -60 <= angle < -15:
        return "forward-left "
    if -120 <= angle < -60:
        return "to your left "
    if -180 <= angle < -120:
        return "to your left rear "
    return ""


def prompt(state, target, instruction) -> str:
    description = instruction.split("degrees from you.")[1].split(" Please control")[0].strip()
    return f"Fly {direction(state, target)}and find the target. {description}\nAction: "


def load_adapter(base, adapter, trainable=False):
    """Attach a LoRA adapter in the AeroRSI layout; the visual merger is kept in FP32 and run under BF16 autocast."""
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    # PEFT copies the BF16 base merger into modules_to_save; reload the adapter's own tensors at full precision.
    restored = 0
    with safe_open(str(Path(adapter) / "adapter_model.safetensors"), framework="pt", device="cpu") as tensors:
        for name, parameter in model.named_parameters():
            if ".visual.merger.modules_to_save." not in name or ".original_module." in name:
                continue
            saved = tensors.get_tensor(name.replace(".modules_to_save.default.", "."))
            parameter.data = parameter.data.to(torch.float32)
            parameter.data.copy_(saved.to(device=parameter.device, dtype=torch.float32))
            restored += parameter.numel()
    if not restored:
        raise ValueError(f"{adapter} has no visual merger: not an AeroRSI/AeroDPO adapter")
    return model


def init_adapter(base_model, output, seed):
    """Write the round-0 adapter: the SFT policy itself, in the trainable layout (every LoRA B is zero)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    torch.manual_seed(seed)
    base = AutoModelForImageTextToText.from_pretrained(base_model, dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(**LORA))
    model.save_pretrained(output, safe_serialization=True)
    return {"base_model": str(base_model), "seed": seed,
            "adapter_sha256": file_hash(Path(output) / "adapter_model.safetensors")}


class Policy:
    def __init__(self, base_model, adapter, attention="sdpa"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        self.processor.image_processor.max_pixels = IMAGE_SIZE * IMAGE_SIZE * 2
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        base = AutoModelForImageTextToText.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto",
                                                           low_cpu_mem_usage=True, trust_remote_code=True,
                                                           attn_implementation=attention)
        self.model = load_adapter(base, adapter).to(self.device).eval()

    def inputs(self, image: Image.Image, text: str):
        import torch
        from qwen_vl_utils import process_vision_info
        if image.size != (IMAGE_SIZE, 2 * IMAGE_SIZE):
            raise ValueError(f"Expected a {IMAGE_SIZE}x{2 * IMAGE_SIZE} mosaic, got {image.size}")
        messages = [[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        images, videos = process_vision_info(messages)
        batch = self.processor(text=texts, images=images, videos=videos, padding="longest", return_tensors="pt")
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        return {k: v.to(self.device) for k, v in batch.items()}

    def generate(self, inputs, *, sample=False, temperature=1.0, seed=None) -> str:
        import torch
        if sample:
            torch.manual_seed(seed)
            options = {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0}
        else:
            options = {"do_sample": False}
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                                    enabled=self.device.type == "cuda"):
            generated = self.model.generate(**inputs, max_new_tokens=20, use_cache=True,
                                            eos_token_id=self.processor.tokenizer.eos_token_id, **options)
        return self.processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def samples(self, inputs, count, seed, temperature):
        """count sampled answers; sample i is drawn after torch.manual_seed(seed + i)."""
        return [self.generate(inputs, sample=True, temperature=temperature, seed=seed + i) for i in range(count)]

    def predict(self, observation, target, instruction) -> Prediction:
        text = self.generate(self.inputs(mosaic(observation), prompt(observation["sensors"]["state"], target, instruction)))
        action, stop, valid = decode(text)
        return Prediction(action, stop, text, valid)
