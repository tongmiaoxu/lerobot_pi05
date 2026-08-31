from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ROOT = Path(__file__).resolve().parent.parent
_SEGMENT_MODELS = None


def get_segment_models(device: str):
    """Lazy-load Grounding-DINO + SAM2 models and cache them per device."""
    global _SEGMENT_MODELS
    if _SEGMENT_MODELS is None or _SEGMENT_MODELS["device"] != device:
        checkpoint = str(ROOT / "weights" / "sam2" / "sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        _SEGMENT_MODELS = {
            "device": device,
            "processor": processor,
            "grounding_model": grounding_model,
            "image_predictor": image_predictor,
        }
    return (
        _SEGMENT_MODELS["processor"],
        _SEGMENT_MODELS["grounding_model"],
        _SEGMENT_MODELS["image_predictor"],
    )


def segment_object_mask(image_bgr, text_prompt: str = "plush toy"):
    """Segment an object in a BGR image using Grounding-DINO + SAM2."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, grounding_model, image_predictor = get_segment_models(device)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    inputs = processor(images=image_pil, text=[text_prompt], return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.325,
        text_threshold=0.3,
        target_sizes=[image_pil.size[::-1]],
    )

    input_boxes = results[0]["boxes"].cpu().numpy()
    if len(input_boxes) == 0:
        return None

    image_predictor.set_image(np.array(image_pil.convert("RGB")))
    masks_list = []
    for box in input_boxes:
        mask, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        mask = mask.squeeze(0)
        masks_list.append(mask.astype(bool))

    if not masks_list:
        return None

    return np.logical_or.reduce(np.stack(masks_list, axis=0), axis=0)


def segment_point_mask(image_bgr, point_xy):
    """Segment one object in a BGR image using a positive SAM2 point prompt."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, image_predictor = get_segment_models(device)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_predictor.set_image(image_rgb)
    mask, _, _ = image_predictor.predict(
        point_coords=np.array([point_xy], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        box=None,
        multimask_output=False,
    )
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    mask = mask.squeeze(0)
    return mask.astype(bool)
