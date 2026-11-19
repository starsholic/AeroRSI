"""Weighted supervised fine-tuning on one round's rows (repairs, normal states, replay).

Each row is an image, a prompt and one target action. The loss of a row is its mean token negative log-likelihood
over the target (including EOS), multiplied by the row's supervision weight. Rows are consumed in the order the
assembly wrote them; a final partial batch is scaled up to count as a full one (and padded with zero-weight
copies), so every row contributes equally across epochs. Training continues the previous round's adapter.
"""
import math
import os
from pathlib import Path

import torch
from PIL import Image

from .policy import IMAGE_SIZE, load_adapter
from .utils import file_hash, read_jsonl


def selective_log_softmax(logits, index):
    """log_softmax(logits).gather(index), row by row to bound memory (as TRL computes it)."""
    if logits.dtype in (torch.float32, torch.float64):
        selected = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        return selected - torch.stack([torch.logsumexp(row, dim=-1) for row in logits])
    return torch.stack([torch.log_softmax(row, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                        for row, labels in zip(logits, index, strict=True)])


def flush_left(mask, *tensors):
    """Shift each row's unmasked tokens to the left and drop all-padding columns (TRL's flush_left)."""
    _, width = mask.shape
    roll = (torch.arange(width).unsqueeze(0) + mask.argmax(dim=1).unsqueeze(1)) % width
    mask = mask.gather(1, roll)
    tensors = [t.gather(1, roll) for t in tensors]
    empty = mask.sum(dim=0) == 0
    end = int(empty.to(torch.int8).argmax()) if empty.any() else width
    return [mask[:, :end]] + [t[:, :end] for t in tensors]


def epoch_plan(size, batch):
    """(indices, tail weights): sequential rows, the last partial batch up-weighted and padded with zero weight."""
    tail = size % batch
    weights = [1.0] * (size - tail) + ([batch / tail] * tail if tail else [])
    padding = (-size) % batch
    return list(range(size)) + [0] * padding, weights + [0.0] * padding


class RowDataset(torch.utils.data.Dataset):
    def __init__(self, rows, processor, batch):
        self.examples = []
        for row in rows:
            if file_hash(row["image"]) != row["image_sha256"] or not row["supervision_weight"] > 0:
                raise ValueError(f"Row {row['id']}: changed image or non-positive weight")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
            self.examples.append({"prompt": processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                                  "target": row["target_text"] + processor.tokenizer.eos_token,
                                  "image": row["image"], "weight": float(row["supervision_weight"])})
        self.indices, self.tail_weights = epoch_plan(len(self.examples), batch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return {**self.examples[self.indices[i]], "tail_weight": self.tail_weights[i]}


class Collator:
    """Prompt (left-padded, with the image) followed by the target (right-padded), then flushed left.

    Images take the same two resize stages as at inference: qwen-vl-utils' fetch_image, then the processor.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        from qwen_vl_utils.vision_process import fetch_image
        images = [[fetch_image({"image": Image.open(e["image"])})] for e in examples]
        prompts = self.processor(images=images, text=[e["prompt"] for e in examples], padding=True, padding_side="left",
                                 return_tensors="pt", add_special_tokens=False)
        targets = self.processor(text=[e["target"] for e in examples], padding=True, padding_side="right",
                                 return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat((prompts["input_ids"], targets["input_ids"]), dim=1)
        attention = torch.cat((prompts["attention_mask"], targets["attention_mask"]), dim=1)
        completion = torch.cat((torch.zeros_like(prompts["attention_mask"]), targets["attention_mask"]), dim=1)
        extra = []
        if "mm_token_type_ids" in prompts:
            extra = [torch.cat((prompts["mm_token_type_ids"], torch.zeros_like(targets["input_ids"])), dim=1)]
        attention, input_ids, completion, *extra = flush_left(attention, input_ids, completion, *extra)
        batch = {"input_ids": input_ids, "attention_mask": attention, "completion_mask": completion,
                 "pixel_values": prompts["pixel_values"], "image_grid_thw": prompts["image_grid_thw"],
                 "tail_weight": torch.tensor([e["tail_weight"] for e in examples], dtype=torch.float32),
                 "weight": torch.tensor([e["weight"] for e in examples], dtype=torch.float32)}
        if extra:
            batch["mm_token_type_ids"] = extra[0]
        return batch


def make_trainer_class():
    from transformers import Trainer

    class WeightedSFTTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            tail, weight = inputs.pop("tail_weight"), inputs.pop("weight")
            mask = inputs.pop("completion_mask")[:, 1:].bool()
            outputs = model(**inputs, use_cache=False)
            logps = selective_log_softmax(outputs.logits[:, :-1, :].contiguous(), inputs["input_ids"][:, 1:].contiguous())
            per_row = -(logps * mask).sum(-1) / mask.sum(-1)
            loss = (per_row * tail * weight).mean()
            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer


def train(config, rows_path, adapter, output):
    """Continue `adapter` on the rows; writes output/final (adapter + processor) and output/metrics."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
    t, model_config = config["train"], config["model"]
    rows = read_jsonl(rows_path)
    torch.manual_seed(config["seed"])
    processor = AutoProcessor.from_pretrained(model_config["base"])
    processor.image_processor.max_pixels = IMAGE_SIZE * IMAGE_SIZE * 2
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    base = AutoModelForImageTextToText.from_pretrained(model_config["base"], dtype=torch.bfloat16,
                                                       attn_implementation=model_config["attention"])
    model = load_adapter(base, adapter, trainable=True)
    for module in model.modules():  # deterministic targets: no dropout
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Training runs as one process on one GPU (global batch 16 by gradient accumulation)")
    if t["global_batch"] % t["per_device_batch"]:
        raise ValueError("The global batch must be a multiple of the per-device batch")
    args = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=t["per_device_batch"],
        gradient_accumulation_steps=t["global_batch"] // t["per_device_batch"],
        learning_rate=t["learning_rate"], num_train_epochs=t["epochs"], max_grad_norm=t["max_grad_norm"],
        lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0, optim="adamw_torch_fused", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config["seed"], data_seed=config["seed"], logging_steps=1, save_strategy="steps",
        save_steps=t["save_steps"], report_to="none", remove_unused_columns=False, ddp_find_unused_parameters=False,
        dataloader_num_workers=t["dataloader_workers"], dataloader_persistent_workers=t["dataloader_workers"] > 0)
    dataset = RowDataset(rows, processor, t["global_batch"])
    trainer = make_trainer_class()(model=model, args=args, train_dataset=dataset, data_collator=Collator(processor),
                                   processing_class=processor)
    trainer.model_accepts_loss_kwargs = False  # average over accumulation steps, not over tokens
    result = trainer.train()
    final = Path(output) / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(final)
    metrics = {**result.metrics, "rows": len(rows), "optimizer_updates": trainer.state.global_step,
               "expected_updates": t["epochs"] * math.ceil(len(rows) / t["global_batch"]),
               "previous_adapter": str(adapter), "rows_sha256": file_hash(rows_path)}
    return metrics
