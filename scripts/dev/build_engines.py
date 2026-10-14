"""Generate engines/*.yaml and workflows/<engine>/{standard.api.json,bindings.yaml}.

Workflows are authored here in a tiny graph DSL (from the official ComfyUI templates' settings), so
node ids, links and bindings stay consistent. Run after editing, then `higgsless workflows validate`.

    uv run python scripts/dev/build_engines.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

REPO = Path(__file__).resolve().parents[2]
LOCK = YAML(typ="safe").load((REPO / "worker" / "models.lock.yaml").read_text(encoding="utf-8"))["models"]

NEG_WAN = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def M(model_id: str) -> str:
    """Filename ComfyUI sees for a lockfile model."""
    entry = LOCK[model_id]
    return entry.get("name") or entry["file"].rsplit("/", 1)[-1]


class Ref:
    def __init__(self, node_id: str) -> None:
        self.id = node_id

    def __getitem__(self, slot: int) -> list:
        return [self.id, slot]


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def add(self, class_type: str, title: str, **inputs: Any) -> Ref:
        node_id = str(len(self.nodes) + 1)
        clean = {}
        for key, value in inputs.items():
            if value is None:
                continue
            clean[key.replace("__", ".")] = value
        self.nodes[node_id] = {"class_type": class_type, "inputs": clean, "_meta": {"title": title}}
        return Ref(node_id)


def P(target: str | list[str], type_: str = "str", **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"target": target, "type": type_}
    out.update(kw)
    return out


# ------------------------------------------------------------------------------------------------
# Shared Wan pieces
# ------------------------------------------------------------------------------------------------
def wan_text(g: Graph, clip_model: str = "umt5_xxl_fp8") -> tuple[Ref, Ref, Ref]:
    clip = g.add("CLIPLoader", "Text Encoder", clip_name=M(clip_model), type="wan", device="default")
    pos = g.add("CLIPTextEncode", "Positive Prompt", clip=clip[0], text="")
    neg = g.add("CLIPTextEncode", "Negative Prompt", clip=clip[0], text=NEG_WAN)
    return clip, pos, neg


def video_in(g: Graph, title: str) -> Ref:
    load = g.add("LoadVideo", title, file="placeholder.mp4")
    return g.add("GetVideoComponents", f"{title} Frames", video=load[0])


def sampler_chain(
    g: Graph,
    model: list,
    positive: list,
    negative: list,
    latent: list,
    *,
    sampler: str,
    steps: int,
    cfg: float,
    title: str = "Sampler",
) -> Ref:
    select = g.add("KSamplerSelect", "Sampler Select", sampler_name=sampler)
    sched = g.add("BasicScheduler", "Scheduler", model=model, scheduler="simple", steps=steps, denoise=1.0)
    return g.add(
        "SamplerCustom",
        title,
        model=model,
        add_noise=True,
        noise_seed=0,
        cfg=cfg,
        positive=positive,
        negative=negative,
        sampler=select[0],
        sigmas=sched[0],
        latent_image=latent,
    )


COMMON_SAMPLING = {
    "seed": P(["Sampler.noise_seed"], "seed"),
    "steps": P("Scheduler.steps", "int"),
    "cfg": P("Sampler.cfg", "float"),
    "sampler": P("Sampler Select.sampler_name"),
    "scheduler": P("Scheduler.scheduler"),
}


# ------------------------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------------------------
def wf_track_sam3() -> tuple[dict, dict]:
    g = Graph()
    frames = video_in(g, "Video")
    sam = g.add("CheckpointLoaderSimple", "SAM3 Model", ckpt_name=M("sam31"))
    text = g.add("CLIPTextEncode", "SAM3 Prompt", clip=sam[1], text="person")
    track = g.add(
        "SAM3_VideoTrack",
        "Track People",
        images=frames[0],
        model=sam[0],
        conditioning=text[0],
        detection_threshold=0.5,
        max_objects=8,
        detect_interval=1,
    )
    colored = g.add(
        "SCAIL2ColoredMask",
        "Colored Masks",
        driving_track_data=track[0],
        object_indices="",
        sort_by="left_to_right",
        replacement_mode=False,
    )
    g.add("SaveImage", "Save Masks", images=colored[0], filename_prefix="masks")
    bindings = {
        "params": {
            "video": P("Video.file", "input_path", required=True, description="Shot clip (relative to the data root)"),
            "text": P("SAM3 Prompt.text", default="person"),
            "threshold": P("Track People.detection_threshold", "float"),
            "max_objects": P("Track People.max_objects", "int"),
            "detect_interval": P("Track People.detect_interval", "int"),
        },
        "outputs": {"masks": {"node": "Save Masks"}},
    }
    return g.nodes, bindings


def wf_caption_qwen3vl() -> tuple[dict, dict]:
    g = Graph()
    image = g.add("LoadImage", "Image", image="placeholder.png")
    clip = g.add("CLIPLoader", "Qwen3-VL", clip_name=M("qwen3vl_4b_fp8"), type="ideogram4", device="default")
    gen = g.add(
        "TextGenerate",
        "Generate Text",
        clip=clip[0],
        prompt="",
        image=image[0],
        max_length=400,
        sampling_mode="off",
        thinking=False,
        use_default_template=True,
    )
    g.add("PreviewAny", "Text Output", source=gen[0])
    bindings = {
        "params": {
            "image": P("Image.image", "input_path", required=True),
            "prompt": P("Generate Text.prompt", required=True),
            "max_length": P("Generate Text.max_length", "int"),
        },
        "outputs": {"text": {"node": "Text Output", "prefix_input": None, "kind": "text"}},
    }
    return g.nodes, bindings


# ------------------------------------------------------------------------------------------------
# Images: character portraits, sheets, keyframes
# ------------------------------------------------------------------------------------------------
def wf_qwen_image21(with_refs: bool) -> tuple[dict, dict]:
    g = Graph()
    unet = g.add("UNETLoader", "Diffusion Model", unet_name=M("qwen_image21_int8"), weight_dtype="default")
    cache = g.add("QwenImage21Cache", "Model Cache", model=unet[0], device="auto", dtype="default")
    clip = g.add("CLIPLoader", "Text Encoder", clip_name=M("qwen3vl_8b_int8"), type="qwen_image", device="default")
    vae = g.add("VAELoader", "VAE", vae_name=M("qwen_image21_vae"))
    refs: dict[str, Any] = {}
    if with_refs:
        for i in (1, 2, 3, 4):
            img = g.add("LoadImage", f"Image {i}", image="placeholder.png")
            refs[f"images__image_{i}"] = img[0]
    enc = g.add(
        "TextEncodeQwenImage21",
        "Encode",
        clip=clip[0],
        prompt="",
        negative_prompt="",
        vae=vae[0],
        resolution=1024,
        **refs,
    )
    latent = g.add("EmptyLatentImage", "Canvas", width=1024, height=1024, batch_size=1)
    ks = g.add(
        "KSampler",
        "Sampler",
        model=cache[0],
        seed=0,
        steps=25,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=enc[0],
        negative=enc[1],
        latent_image=latent[0],
        denoise=1.0,
    )
    dec = g.add("VAEDecode", "Decode", samples=ks[0], vae=vae[0])
    g.add("SaveImage", "Save Image", images=dec[0], filename_prefix="image")
    params: dict[str, Any] = {
        "prompt": P("Encode.prompt", required=True),
        "negative": P("Encode.negative_prompt", default=""),
        "width": P("Canvas.width", "int"),
        "height": P("Canvas.height", "int"),
        "resolution": P("Encode.resolution", "int"),
        "seed": P("Sampler.seed", "seed"),
        "steps": P("Sampler.steps", "int"),
        "cfg": P("Sampler.cfg", "float"),
        "unet": P("Diffusion Model.unet_name"),
    }
    bypass: dict[str, Any] = {"Model Cache": {0: "model"}}
    params["cache"] = P("Model Cache.device", optional_node="Model Cache", default="auto")
    if with_refs:
        params["image_1"] = P(
            "Image 1.image",
            "input_path",
            required=True,
            description="First reference; also sets the canvas size unless width/height are set",
        )
        for i in (2, 3, 4):
            params[f"image_{i}"] = P(f"Image {i}.image", "input_path", optional_node=f"Image {i}")
            bypass[f"Image {i}"] = "remove"
    return g.nodes, {"params": params, "bypass": bypass, "outputs": {"image": {"node": "Save Image"}}}


def wf_keyframe_qwen_edit() -> tuple[dict, dict]:
    g = Graph()
    unet = g.add("UNETLoader", "Diffusion Model", unet_name=M("qwen_edit2511_int8"), weight_dtype="default")
    shift = g.add("ModelSamplingAuraFlow", "Model Sampling", model=unet[0], shift=3.1)
    norm = g.add("CFGNorm", "CFG Norm", model=shift[0], strength=1.0)
    angle = g.add(
        "LoraLoaderModelOnly", "Angle LoRA", model=norm[0], lora_name=M("qwen_edit_multiangle"), strength_model=1.0
    )
    fast = g.add(
        "LoraLoaderModelOnly", "Lightning LoRA", model=angle[0], lora_name=M("qwen_edit_lightning"), strength_model=1.0
    )
    clip = g.add("CLIPLoader", "Text Encoder", clip_name=M("qwen25_vl_7b_fp8"), type="qwen_image", device="default")
    vae = g.add("VAELoader", "VAE", vae_name=M("qwen_image_vae"))
    base = g.add("LoadImage", "Base Image", image="placeholder.png")
    scaled = g.add("FluxKontextImageScale", "Scale Base", image=base[0])
    ref2 = g.add("LoadImage", "Reference 2", image="placeholder.png")
    ref3 = g.add("LoadImage", "Reference 3", image="placeholder.png")
    pos = g.add(
        "TextEncodeQwenImageEditPlus",
        "Positive",
        clip=clip[0],
        prompt="",
        vae=vae[0],
        image1=scaled[0],
        image2=ref2[0],
        image3=ref3[0],
    )
    neg = g.add(
        "TextEncodeQwenImageEditPlus",
        "Negative",
        clip=clip[0],
        prompt="",
        vae=vae[0],
        image1=scaled[0],
        image2=ref2[0],
        image3=ref3[0],
    )
    pos_m = g.add(
        "FluxKontextMultiReferenceLatentMethod",
        "Positive Refs",
        conditioning=pos[0],
        reference_latents_method="index_timestep_zero",
    )
    neg_m = g.add(
        "FluxKontextMultiReferenceLatentMethod",
        "Negative Refs",
        conditioning=neg[0],
        reference_latents_method="index_timestep_zero",
    )
    latent = g.add("VAEEncode", "Encode Base", pixels=scaled[0], vae=vae[0])
    ks = g.add(
        "KSampler",
        "Sampler",
        model=fast[0],
        seed=0,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=pos_m[0],
        negative=neg_m[0],
        latent_image=latent[0],
        denoise=1.0,
    )
    dec = g.add("VAEDecode", "Decode", samples=ks[0], vae=vae[0])
    g.add("SaveImage", "Save Image", images=dec[0], filename_prefix="keyframe")
    params = {
        "base_image": P(
            "Base Image.image",
            "input_path",
            required=True,
            description="Shot frame (keyframe) or character image (sheet/angles)",
        ),
        "reference_2": P("Reference 2.image", "input_path", optional_node="Reference 2"),
        "reference_3": P("Reference 3.image", "input_path", optional_node="Reference 3"),
        "prompt": P("Positive.prompt", required=True),
        "negative": P("Negative.prompt", default=""),
        "seed": P("Sampler.seed", "seed"),
        "steps": P("Sampler.steps", "int"),
        "cfg": P("Sampler.cfg", "float"),
        "lightning_lora": P("Lightning LoRA.lora_name", optional_node="Lightning LoRA"),
        "angle_lora": P("Angle LoRA.lora_name", optional_node="Angle LoRA"),
        "unet": P("Diffusion Model.unet_name"),
    }
    bypass = {
        "Lightning LoRA": {0: "model"},
        "Angle LoRA": {0: "model"},
        "Reference 2": "remove",
        "Reference 3": "remove",
    }
    return g.nodes, {"params": params, "bypass": bypass, "outputs": {"image": {"node": "Save Image"}}}


# ------------------------------------------------------------------------------------------------
# Video: rebuild (Wan-Animate-2), SCAIL-2, Wan2.2-Animate Mix, Bernini-R, VACE-Fun
# ------------------------------------------------------------------------------------------------
def wf_wan_animate2() -> tuple[dict, dict]:
    g = Graph()
    unet = g.add("UNETLoader", "Diffusion Model", unet_name=M("wan_animate2_int8"), weight_dtype="default")
    speed = g.add(
        "LoraLoaderModelOnly", "Speed LoRA", model=unet[0], lora_name=M("lightx2v_i2v_480p"), strength_model=1.0
    )
    cache = g.add("WanAnimate2Cache", "Pose Cache", model=speed[0], device="gpu", dtype="int8")
    shift = g.add("ModelSamplingSD3", "Model Sampling", model=cache[0], shift=5.0)
    clip, pos, neg = wan_text(g)
    pose_prompt = g.add(
        "CLIPTextEncode", "Pose Prompt", clip=clip[0], text="a person performing, background stationary"
    )
    vae = g.add("VAELoader", "VAE", vae_name=M("wan21_vae_bf16"))
    vision = g.add("CLIPVisionLoader", "CLIP Vision", clip_name=M("clip_vision_h"))
    ref = g.add("LoadImage", "Reference Image", image="placeholder.png")
    drive = video_in(g, "Driving Video")
    first = g.add("ImageFromBatch", "Driving First Frame", image=drive[0], batch_index=0, length=1)
    ref_vision = g.add("CLIPVisionEncode", "Reference Vision", clip_vision=vision[0], image=ref[0], crop="none")
    pose_vision = g.add("CLIPVisionEncode", "Pose Vision", clip_vision=vision[0], image=first[0], crop="none")
    cont = video_in(g, "Continue Frames")
    anim = g.add(
        "WanAnimate2ToVideo",
        "Animate",
        positive=pos[0],
        negative=neg[0],
        vae=vae[0],
        width=832,
        height=480,
        length=81,
        batch_size=1,
        reference_image=ref[0],
        pose_video=drive[0],
        clip_vision_output=ref_vision[0],
        positive_pose=pose_prompt[0],
        clip_vision_output_pose=pose_vision[0],
        continue_motion=cont[0],
        video_frame_offset=0,
        pose_strength=1.0,
        pose_start_percent=0.0,
        pose_end_percent=1.0,
        reference_image_strength=1.0,
    )
    sample = sampler_chain(g, shift[0], anim[0], anim[1], anim[2], sampler="lcm", steps=6, cfg=1.0)
    trim = g.add("TrimVideoLatent", "Trim Latent", samples=sample[0], trim_amount=anim[3])
    dec = g.add("VAEDecode", "Decode", samples=trim[0], vae=vae[0])
    keep = g.add("ImageFromBatch", "Drop Overlap", image=dec[0], batch_index=anim[4], length=4096)
    g.add("SaveImage", "Save Frames", images=keep[0], filename_prefix="frames")
    params = {
        "reference_image": P("Reference Image.image", "input_path", required=True),
        "driving_video": P("Driving Video.file", "input_path", required=True),
        "continue_video": P(
            "Continue Frames.file",
            "input_path",
            optional_node=["Continue Frames Frames", "Continue Frames"],
            description="Previous window's output tail (continuation windows only)",
        ),
        "prompt": P(
            "Positive Prompt.text",
            required=True,
            description="Character Description: ...\\nBackground description: ... (describes the reference)",
        ),
        "pose_prompt": P("Pose Prompt.text"),
        "negative": P("Negative Prompt.text"),
        "width": P("Animate.width", "int", required=True),
        "height": P("Animate.height", "int", required=True),
        "length": P("Animate.length", "int", required=True),
        "video_frame_offset": P("Animate.video_frame_offset", "int"),
        "reference_strength": P("Animate.reference_image_strength", "float"),
        "pose_strength": P("Animate.pose_strength", "float"),
        "shift": P("Model Sampling.shift", "float"),
        "unet": P("Diffusion Model.unet_name"),
        "speed_lora": P("Speed LoRA.lora_name", optional_node="Speed LoRA"),
        "pose_cache": P("Pose Cache.device", optional_node="Pose Cache"),
        **COMMON_SAMPLING,
    }
    bypass = {
        "Speed LoRA": {0: "model"},
        "Pose Cache": {0: "model"},
        "Continue Frames": "remove",
        "Continue Frames Frames": "remove",
    }
    return g.nodes, {"params": params, "bypass": bypass, "outputs": {"frames": {"node": "Save Frames"}}}


def wf_scail2(replacement: bool) -> tuple[dict, dict]:
    g = Graph()
    unet = g.add("UNETLoader", "Diffusion Model", unet_name=M("scail2_nvfp4"), weight_dtype="default")
    relight = g.add(
        "LoraLoaderModelOnly", "Relight LoRA", model=unet[0], lora_name=M("scail2_relight"), strength_model=1.0
    )
    dpo = g.add("LoraLoaderModelOnly", "DPO LoRA", model=relight[0], lora_name=M("scail2_dpo"), strength_model=1.0)
    speed = g.add(
        "LoraLoaderModelOnly", "Speed LoRA", model=dpo[0], lora_name=M("lightx2v_i2v_480p"), strength_model=0.8
    )
    shift = g.add("ModelSamplingSD3", "Model Sampling", model=speed[0], shift=5.0)
    _, pos, neg = wan_text(g)
    vae = g.add("VAELoader", "VAE", vae_name=M("wan21_vae_bf16"))
    vision = g.add("CLIPVisionLoader", "CLIP Vision", clip_name=M("clip_vision_h"))
    sam = g.add("CheckpointLoaderSimple", "SAM3 Model", ckpt_name=M("sam31"))
    sam_text = g.add("CLIPTextEncode", "SAM3 Prompt", clip=sam[1], text="person")
    drive = video_in(g, "Driving Video")
    ref = g.add("LoadImage", "Reference Image", image="placeholder.png")
    track_d = g.add(
        "SAM3_VideoTrack",
        "Track Driving",
        images=drive[0],
        model=sam[0],
        conditioning=sam_text[0],
        detection_threshold=0.5,
        max_objects=4,
        detect_interval=1,
    )
    track_r = g.add(
        "SAM3_VideoTrack",
        "Track Reference",
        images=ref[0],
        model=sam[0],
        conditioning=sam_text[0],
        detection_threshold=0.5,
        max_objects=4,
        detect_interval=1,
    )
    masks = g.add(
        "SCAIL2ColoredMask",
        "Colored Masks",
        driving_track_data=track_d[0],
        ref_track_data=track_r[0],
        object_indices="",
        sort_by="left_to_right",
        replacement_mode=replacement,
    )
    ref_vision = g.add("CLIPVisionEncode", "Reference Vision", clip_vision=vision[0], image=ref[0], crop="none")
    prev = video_in(g, "Previous Frames")
    scail = g.add(
        "WanSCAILToVideo",
        "SCAIL",
        positive=pos[0],
        negative=neg[0],
        vae=vae[0],
        width=896,
        height=512,
        length=81,
        batch_size=1,
        pose_video=drive[0],
        pose_video_mask=masks[0],
        replacement_mode=replacement,
        pose_strength=1.0,
        pose_start=0.0,
        pose_end=1.0,
        reference_image=ref[0],
        reference_image_mask=masks[1],
        clip_vision_output=ref_vision[0],
        video_frame_offset=0,
        previous_frame_count=5,
        previous_frames=prev[0],
    )
    sample = sampler_chain(g, shift[0], scail[0], scail[1], scail[2], sampler="euler", steps=6, cfg=1.0)
    dec = g.add("VAEDecode", "Decode", samples=sample[0], vae=vae[0])
    keep = g.add("ImageFromBatch", "Drop Overlap", image=dec[0], batch_index=0, length=4096)
    g.add("SaveImage", "Save Frames", images=keep[0], filename_prefix="frames")
    params = {
        "reference_image": P(
            "Reference Image.image",
            "input_path",
            required=True,
            description="Composite keyframe with every new character in place",
        ),
        "driving_video": P("Driving Video.file", "input_path", required=True),
        "previous_video": P(
            "Previous Frames.file", "input_path", optional_node=["Previous Frames Frames", "Previous Frames"]
        ),
        "skip_frames": P("Drop Overlap.batch_index", "int", description="Frames re-decoded from the previous window"),
        "prompt": P("Positive Prompt.text", required=True),
        "negative": P("Negative Prompt.text"),
        "track_text": P("SAM3 Prompt.text"),
        "max_people": P(["Track Driving.max_objects", "Track Reference.max_objects"], "int"),
        "width": P("SCAIL.width", "int", required=True),
        "height": P("SCAIL.height", "int", required=True),
        "length": P("SCAIL.length", "int", required=True),
        "video_frame_offset": P("SCAIL.video_frame_offset", "int"),
        "pose_strength": P("SCAIL.pose_strength", "float"),
        "shift": P("Model Sampling.shift", "float"),
        "unet": P("Diffusion Model.unet_name"),
        "relight_lora": P("Relight LoRA.lora_name", optional_node="Relight LoRA"),
        "dpo_lora": P("DPO LoRA.lora_name", optional_node="DPO LoRA"),
        "speed_lora": P("Speed LoRA.lora_name", optional_node="Speed LoRA"),
        **COMMON_SAMPLING,
    }
    bypass = {
        "Relight LoRA": {0: "model"},
        "DPO LoRA": {0: "model"},
        "Speed LoRA": {0: "model"},
        "Previous Frames": "remove",
        "Previous Frames Frames": "remove",
    }
    return g.nodes, {"params": params, "bypass": bypass, "outputs": {"frames": {"node": "Save Frames"}}}


def wf_wan22_animate_mix() -> tuple[dict, dict]:
    g = Graph()
    unet = g.add("UNETLoader", "Diffusion Model", unet_name=M("wan22_animate_int8"), weight_dtype="default")
    relight = g.add(
        "LoraLoaderModelOnly", "Relight LoRA", model=unet[0], lora_name=M("wan22_animate_relight"), strength_model=1.0
    )
    speed = g.add(
        "LoraLoaderModelOnly", "Speed LoRA", model=relight[0], lora_name=M("lightx2v_i2v_480p"), strength_model=1.0
    )
    shift = g.add("ModelSamplingSD3", "Model Sampling", model=speed[0], shift=5.0)
    _, pos, neg = wan_text(g)
    vae = g.add("VAELoader", "VAE", vae_name=M("wan21_vae"))
    vision = g.add("CLIPVisionLoader", "CLIP Vision", clip_name=M("clip_vision_h"))
    ref = g.add("LoadImage", "Reference Image", image="placeholder.png")
    drive = video_in(g, "Driving Video")
    pose_model = g.add(
        "OnnxDetectionModelLoader",
        "Pose Model",
        vitpose_model=M("vitpose_l"),
        yolo_model=M("yolov10m"),
        onnx_device="CUDAExecutionProvider",
    )
    detect = g.add("PoseAndFaceDetection", "Pose And Face", model=pose_model[0], images=drive[0], width=832, height=480)
    pose_img = g.add(
        "DrawViTPose",
        "Draw Pose",
        pose_data=detect[0],
        width=832,
        height=480,
        retarget_padding=16,
        body_stick_width=-1,
        hand_stick_width=-1,
        draw_head=True,
    )
    sam = g.add("CheckpointLoaderSimple", "SAM3 Model", ckpt_name=M("sam31"))
    sam_text = g.add("CLIPTextEncode", "SAM3 Prompt", clip=sam[1], text="person")
    track = g.add(
        "SAM3_VideoTrack",
        "Track Driving",
        images=drive[0],
        model=sam[0],
        conditioning=sam_text[0],
        detection_threshold=0.5,
        max_objects=4,
        detect_interval=1,
    )
    pick = g.add("SAM3_TrackToMask", "Pick Performer", track_data=track[0], object_indices="0")
    grow = g.add("GrowMask", "Grow Mask", mask=pick[0], expand=16, tapered_corners=True)
    ref_vision = g.add("CLIPVisionEncode", "Reference Vision", clip_vision=vision[0], image=ref[0], crop="none")
    cont = video_in(g, "Continue Frames")
    anim = g.add(
        "WanAnimateToVideo",
        "Animate",
        positive=pos[0],
        negative=neg[0],
        vae=vae[0],
        width=832,
        height=480,
        length=77,
        batch_size=1,
        clip_vision_output=ref_vision[0],
        reference_image=ref[0],
        face_video=detect[1],
        pose_video=pose_img[0],
        continue_motion_max_frames=5,
        background_video=drive[0],
        character_mask=grow[0],
        continue_motion=cont[0],
        video_frame_offset=0,
    )
    sample = sampler_chain(g, shift[0], anim[0], anim[1], anim[2], sampler="euler", steps=6, cfg=1.0)
    trim = g.add("TrimVideoLatent", "Trim Latent", samples=sample[0], trim_amount=anim[3])
    dec = g.add("VAEDecode", "Decode", samples=trim[0], vae=vae[0])
    keep = g.add("ImageFromBatch", "Drop Overlap", image=dec[0], batch_index=anim[4], length=4096)
    g.add("SaveImage", "Save Frames", images=keep[0], filename_prefix="frames")
    params = {
        "reference_image": P("Reference Image.image", "input_path", required=True),
        "driving_video": P("Driving Video.file", "input_path", required=True),
        "continue_video": P(
            "Continue Frames.file", "input_path", optional_node=["Continue Frames Frames", "Continue Frames"]
        ),
        "performer_index": P(
            "Pick Performer.object_indices", description="SAM3 object index (left to right) to replace"
        ),
        "mask_grow": P("Grow Mask.expand", "int"),
        "prompt": P("Positive Prompt.text", required=True),
        "negative": P("Negative Prompt.text"),
        "width": P(["Animate.width", "Pose And Face.width", "Draw Pose.width"], "int", required=True),
        "height": P(["Animate.height", "Pose And Face.height", "Draw Pose.height"], "int", required=True),
        "length": P("Animate.length", "int", required=True),
        "video_frame_offset": P("Animate.video_frame_offset", "int"),
        "shift": P("Model Sampling.shift", "float"),
        "unet": P("Diffusion Model.unet_name"),
        "relight_lora": P("Relight LoRA.lora_name", optional_node="Relight LoRA"),
        "speed_lora": P("Speed LoRA.lora_name", optional_node="Speed LoRA"),
        **COMMON_SAMPLING,
    }
    bypass = {
        "Relight LoRA": {0: "model"},
        "Speed LoRA": {0: "model"},
        "Continue Frames": "remove",
        "Continue Frames Frames": "remove",
    }
    return g.nodes, {"params": params, "bypass": bypass, "outputs": {"frames": {"node": "Save Frames"}}}


def wf_bernini(with_refs: bool) -> tuple[dict, dict]:
    g = Graph()
    high = g.add("UNETLoader", "High Noise Model", unet_name=M("bernini_high_int8"), weight_dtype="default")
    low = g.add("UNETLoader", "Low Noise Model", unet_name=M("bernini_low_int8"), weight_dtype="default")
    high_l = g.add(
        "LoraLoaderModelOnly", "Speed LoRA High", model=high[0], lora_name=M("lightx2v_t2v_v2"), strength_model=3.0
    )
    low_l = g.add(
        "LoraLoaderModelOnly", "Speed LoRA Low", model=low[0], lora_name=M("lightx2v_t2v_v2"), strength_model=1.5
    )
    _, pos, neg = wan_text(g)
    vae = g.add("VAELoader", "VAE", vae_name=M("wan21_vae_bf16"))
    src = video_in(g, "Source Video")
    refs: dict[str, Any] = {}
    if with_refs:
        for i in range(3):
            img = g.add("LoadImage", f"Reference {i}", image="placeholder.png")
            refs[f"reference_images__reference_image_{i}"] = img[0]
    cond = g.add(
        "BerniniConditioning",
        "Bernini",
        positive=pos[0],
        negative=neg[0],
        vae=vae[0],
        width=832,
        height=480,
        length=81,
        batch_size=1,
        source_video=src[0],
        ref_max_size=848,
        **refs,
    )
    sched = g.add("BasicScheduler", "Scheduler", model=high_l[0], scheduler="simple", steps=6, denoise=1.0)
    split = g.add("SplitSigmas", "Split Sigmas", sigmas=sched[0], step=3)
    select = g.add("KSamplerSelect", "Sampler Select", sampler_name="res_multistep")
    s1 = g.add(
        "SamplerCustom",
        "Sampler",
        model=high_l[0],
        add_noise=True,
        noise_seed=0,
        cfg=1.0,
        positive=cond[0],
        negative=cond[1],
        sampler=select[0],
        sigmas=split[0],
        latent_image=cond[2],
    )
    s2 = g.add(
        "SamplerCustom",
        "Sampler Low",
        model=low_l[0],
        add_noise=False,
        noise_seed=0,
        cfg=1.0,
        positive=cond[0],
        negative=cond[1],
        sampler=select[0],
        sigmas=split[1],
        latent_image=s1[0],
    )
    dec = g.add("VAEDecode", "Decode", samples=s2[0], vae=vae[0])
    g.add("SaveImage", "Save Frames", images=dec[0], filename_prefix="frames")
    params: dict[str, Any] = {
        "source_video": P("Source Video.file", "input_path", required=True),
        "prompt": P(
            "Positive Prompt.text", required=True, description="Edit instruction; refer to references as image0, image1"
        ),
        "negative": P("Negative Prompt.text"),
        "width": P("Bernini.width", "int", required=True),
        "height": P("Bernini.height", "int", required=True),
        "length": P("Bernini.length", "int", required=True),
        "split_step": P("Split Sigmas.step", "int"),
        "speed_lora_high": P("Speed LoRA High.lora_name", optional_node="Speed LoRA High"),
        "speed_lora_low": P("Speed LoRA Low.lora_name", optional_node="Speed LoRA Low"),
        **COMMON_SAMPLING,
    }
    params["cfg"] = P(["Sampler.cfg", "Sampler Low.cfg"], "float")
    bypass: dict[str, Any] = {"Speed LoRA High": {0: "model"}, "Speed LoRA Low": {0: "model"}}
    if with_refs:
        params["reference_0"] = P("Reference 0.image", "input_path", required=True)
        for i in (1, 2):
            params[f"reference_{i}"] = P(f"Reference {i}.image", "input_path", optional_node=f"Reference {i}")
            bypass[f"Reference {i}"] = "remove"
    return g.nodes, {"params": params, "bypass": bypass, "outputs": {"frames": {"node": "Save Frames"}}}


def wf_vace_fun(inpaint: bool) -> tuple[dict, dict]:
    """Structure-locked restyle (depth control + styled reference) or clean-plate inpainting (masks)."""
    g = Graph()
    high = g.add("UNETLoader", "High Noise Model", unet_name=M("vace_fun_high_fp8"), weight_dtype="default")
    low = g.add("UNETLoader", "Low Noise Model", unet_name=M("vace_fun_low_fp8"), weight_dtype="default")
    high_l = g.add(
        "LoraLoaderModelOnly", "Speed LoRA High", model=high[0], lora_name=M("lightx2v_t2v_high"), strength_model=1.0
    )
    low_l = g.add(
        "LoraLoaderModelOnly", "Speed LoRA Low", model=low[0], lora_name=M("lightx2v_t2v_low"), strength_model=1.0
    )
    hs = g.add("ModelSamplingSD3", "Model Sampling High", model=high_l[0], shift=8.0)
    ls = g.add("ModelSamplingSD3", "Model Sampling Low", model=low_l[0], shift=8.0)
    _, pos, neg = wan_text(g)
    vae = g.add("VAELoader", "VAE", vae_name=M("wan21_vae"))
    src = video_in(g, "Source Video")
    extra: dict[str, Any] = {}
    if inpaint:
        mask_frames = video_in(g, "Mask Video")
        mask = g.add("ImageToMask", "Mask", image=mask_frames[0], channel="red")
        extra = {"control_video": src[0], "control_masks": mask[0]}
    else:
        da3 = g.add("LoadDA3Model", "Depth Model", model_name=M("da3_small"), weight_dtype="default")
        geo = g.add(
            "DA3Inference",
            "Depth",
            da3_model=da3[0],
            image=src[0],
            resolution=504,
            resize_method="upper_bound_resize",
            mode="mono",
        )
        depth = g.add(
            "DA3Render",
            "Depth Render",
            da3_geometry=geo[0],
            output="depth",
            output__normalization="v2_style",
            output__apply_sky_clip=False,
        )
        ref = g.add("LoadImage", "Style Reference", image="placeholder.png")
        extra = {"control_video": depth[0], "reference_image": ref[0]}
    vace = g.add(
        "WanVaceToVideo",
        "VACE",
        positive=pos[0],
        negative=neg[0],
        vae=vae[0],
        width=832,
        height=480,
        length=81,
        batch_size=1,
        strength=1.0,
        **extra,
    )
    sched = g.add("BasicScheduler", "Scheduler", model=hs[0], scheduler="simple", steps=6, denoise=1.0)
    split = g.add("SplitSigmas", "Split Sigmas", sigmas=sched[0], step=3)
    select = g.add("KSamplerSelect", "Sampler Select", sampler_name="euler")
    s1 = g.add(
        "SamplerCustom",
        "Sampler",
        model=hs[0],
        add_noise=True,
        noise_seed=0,
        cfg=1.0,
        positive=vace[0],
        negative=vace[1],
        sampler=select[0],
        sigmas=split[0],
        latent_image=vace[2],
    )
    s2 = g.add(
        "SamplerCustom",
        "Sampler Low",
        model=ls[0],
        add_noise=False,
        noise_seed=0,
        cfg=1.0,
        positive=vace[0],
        negative=vace[1],
        sampler=select[0],
        sigmas=split[1],
        latent_image=s1[0],
    )
    trim = g.add("TrimVideoLatent", "Trim Latent", samples=s2[0], trim_amount=vace[3])
    dec = g.add("VAEDecode", "Decode", samples=trim[0], vae=vae[0])
    g.add("SaveImage", "Save Frames", images=dec[0], filename_prefix="frames")
    params: dict[str, Any] = {
        "source_video": P("Source Video.file", "input_path", required=True),
        "prompt": P("Positive Prompt.text", required=True),
        "negative": P("Negative Prompt.text"),
        "width": P("VACE.width", "int", required=True),
        "height": P("VACE.height", "int", required=True),
        "length": P("VACE.length", "int", required=True),
        "strength": P("VACE.strength", "float"),
        "split_step": P("Split Sigmas.step", "int"),
        **COMMON_SAMPLING,
    }
    params["cfg"] = P(["Sampler.cfg", "Sampler Low.cfg"], "float")
    if inpaint:
        params["mask_video"] = P(
            "Mask Video.file",
            "input_path",
            required=True,
            description="White where the original performer must be removed",
        )
    else:
        params["style_reference"] = P("Style Reference.image", "input_path", required=True)
    return g.nodes, {"params": params, "outputs": {"frames": {"node": "Save Frames"}}}


# ------------------------------------------------------------------------------------------------
# Lip-sync, upscale, interpolation
# ------------------------------------------------------------------------------------------------
def wf_latentsync() -> tuple[dict, dict]:
    g = Graph()
    frames = video_in(g, "Video")
    audio = g.add("LoadAudio", "Vocal", audio="placeholder.wav")
    sync = g.add(
        "LatentSyncNode",
        "LatentSync",
        images=frames[0],
        audio=audio[0],
        seed=1247,
        lips_expression=1.5,
        inference_steps=20,
    )
    g.add("SaveImage", "Save Frames", images=sync[0], filename_prefix="frames")
    params = {
        "video": P("Video.file", "input_path", required=True),
        "audio": P("Vocal.audio", "input_path", required=True, description="Dry vocal slice for exactly this shot"),
        "seed": P("LatentSync.seed", "seed"),
        "lips_expression": P("LatentSync.lips_expression", "float"),
        "steps": P("LatentSync.inference_steps", "int"),
    }
    return g.nodes, {"params": params, "outputs": {"frames": {"node": "Save Frames"}}}


def wf_infinitetalk() -> tuple[dict, dict]:
    g = Graph()
    unet = g.add("UNETLoader", "Diffusion Model", unet_name=M("wan21_i2v_480p_fp8"), weight_dtype="default")
    speed = g.add(
        "LoraLoaderModelOnly", "Speed LoRA", model=unet[0], lora_name=M("lightx2v_i2v_480p"), strength_model=1.0
    )
    patch = g.add("ModelPatchLoader", "InfiniteTalk Patch", name=M("infinitetalk_single"))
    _, pos, neg = wan_text(g)
    vae = g.add("VAELoader", "VAE", vae_name=M("wan21_vae_bf16"))
    vision = g.add("CLIPVisionLoader", "CLIP Vision", clip_name=M("clip_vision_h"))
    start = g.add("LoadImage", "Start Image", image="placeholder.png")
    start_vision = g.add("CLIPVisionEncode", "Start Vision", clip_vision=vision[0], image=start[0], crop="none")
    enc_model = g.add("AudioEncoderLoader", "Audio Encoder", audio_encoder_name=M("wav2vec2_chinese_base"))
    audio = g.add("LoadAudio", "Vocal", audio="placeholder.wav")
    enc = g.add("AudioEncoderEncode", "Encode Audio", audio_encoder=enc_model[0], audio=audio[0])
    prev = video_in(g, "Previous Frames")
    talk = g.add(
        "WanInfiniteTalkToVideo",
        "InfiniteTalk",
        mode="single_speaker",
        model=speed[0],
        model_patch=patch[0],
        positive=pos[0],
        negative=neg[0],
        vae=vae[0],
        width=832,
        height=480,
        length=81,
        clip_vision_output=start_vision[0],
        start_image=start[0],
        audio_encoder_output_1=enc[0],
        motion_frame_count=9,
        audio_scale=1.0,
        previous_frames=prev[0],
    )
    shift = g.add("ModelSamplingSD3", "Model Sampling", model=talk[0], shift=7.0)
    sample = sampler_chain(g, shift[0], talk[1], talk[2], talk[3], sampler="lcm", steps=6, cfg=1.0)
    dec = g.add("VAEDecode", "Decode", samples=sample[0], vae=vae[0])
    keep = g.add("ImageFromBatch", "Drop Overlap", image=dec[0], batch_index=talk[4], length=4096)
    g.add("SaveImage", "Save Frames", images=keep[0], filename_prefix="frames")
    params = {
        "start_image": P(
            "Start Image.image",
            "input_path",
            required=True,
            description="First frame of the rendered shot (the take being lip-synced)",
        ),
        "audio": P("Vocal.audio", "input_path", required=True),
        "previous_video": P(
            "Previous Frames.file", "input_path", optional_node=["Previous Frames Frames", "Previous Frames"]
        ),
        "prompt": P("Positive Prompt.text", required=True),
        "negative": P("Negative Prompt.text"),
        "width": P("InfiniteTalk.width", "int", required=True),
        "height": P("InfiniteTalk.height", "int", required=True),
        "length": P("InfiniteTalk.length", "int", required=True),
        "audio_scale": P("InfiniteTalk.audio_scale", "float"),
        "speed_lora": P("Speed LoRA.lora_name", optional_node="Speed LoRA"),
        **COMMON_SAMPLING,
    }
    bypass = {"Speed LoRA": {0: "model"}, "Previous Frames": "remove", "Previous Frames Frames": "remove"}
    return g.nodes, {"params": params, "bypass": bypass, "outputs": {"frames": {"node": "Save Frames"}}}


def wf_seedvr2() -> tuple[dict, dict]:
    g = Graph()
    unet = g.add("UNETLoader", "Diffusion Model", unet_name=M("seedvr2_7b_nvfp4"), weight_dtype="default")
    vae = g.add("VAELoader", "VAE", vae_name=M("seedvr2_vae"))
    frames = video_in(g, "Video")
    scaled = g.add(
        "ImageScale", "Resize", image=frames[0], upscale_method="lanczos", width=1920, height=1080, crop="disabled"
    )
    prep = g.add("SeedVR2Preprocess", "Preprocess", resized_images=scaled[0])
    enc = g.add(
        "VAEEncodeTiled",
        "Encode",
        pixels=prep[0],
        vae=vae[0],
        tile_size=512,
        overlap=128,
        temporal_size=64,
        temporal_overlap=8,
    )
    cond = g.add("SeedVR2Conditioning", "Conditioning", model=unet[0], vae_conditioning=enc[0])
    ks = g.add(
        "KSampler",
        "Sampler",
        model=unet[0],
        seed=0,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=cond[0],
        negative=cond[1],
        latent_image=enc[0],
        denoise=1.0,
    )
    dec = g.add(
        "VAEDecodeTiled",
        "Decode",
        samples=ks[0],
        vae=vae[0],
        tile_size=512,
        overlap=128,
        temporal_size=64,
        temporal_overlap=8,
    )
    post = g.add(
        "SeedVR2PostProcessing", "Post", images=dec[0], original_resized_images=scaled[0], color_correction_method="lab"
    )
    g.add("SaveImage", "Save Frames", images=post[0], filename_prefix="frames")
    params = {
        "video": P("Video.file", "input_path", required=True),
        "width": P("Resize.width", "int", required=True),
        "height": P("Resize.height", "int", required=True),
        "unet": P("Diffusion Model.unet_name"),
        "color_correction": P("Post.color_correction_method"),
        "seed": P("Sampler.seed", "seed"),
    }
    return g.nodes, {"params": params, "outputs": {"frames": {"node": "Save Frames"}}}


def wf_interp(model_id: str) -> tuple[dict, dict]:
    g = Graph()
    frames = video_in(g, "Video")
    loader = g.add("FrameInterpolationModelLoader", "Interpolation Model", model_name=M(model_id))
    interp = g.add("FrameInterpolate", "Interpolate", interp_model=loader[0], images=frames[0], multiplier=2)
    g.add("SaveImage", "Save Frames", images=interp[0], filename_prefix="frames")
    params = {
        "video": P("Video.file", "input_path", required=True),
        "multiplier": P("Interpolate.multiplier", "int", required=True),
    }
    return g.nodes, {"params": params, "outputs": {"frames": {"node": "Save Frames"}}}


# ------------------------------------------------------------------------------------------------
# Engine manifests
# ------------------------------------------------------------------------------------------------
APACHE = {"weights": "Apache-2.0", "code": "Apache-2.0", "commercial": True}
MIT = {"weights": "MIT", "code": "MIT", "commercial": True}
VIDEO_PROFILES_A2 = {
    "draft": {},
    "standard": {},
    "max": {"unet": M("wan_animate2_bf16"), "speed_lora": None, "steps": 40, "cfg": 3.0, "sampler": "euler"},
}

ENGINES: list[dict[str, Any]] = [
    {
        "id": "track.sam3",
        "title": "SAM 3.1 person tracking (colored identity masks)",
        "capability": "track",
        "workflow": wf_track_sam3,
        "models": ["sam31"],
        "licence": {
            "weights": "SAM License",
            "code": "Apache-2.0",
            "commercial": True,
            "notes": "Meta SAM License: commercial use allowed with its use restrictions",
        },
        "default": True,
        "defaults": {"text": "person", "threshold": 0.5, "max_objects": 8, "detect_interval": 1},
    },
    {
        "id": "caption.qwen3vl",
        "title": "Qwen3-VL captions",
        "capability": "caption",
        "workflow": wf_caption_qwen3vl,
        "models": ["qwen3vl_4b_fp8"],
        "licence": APACHE,
        "default": True,
        "defaults": {"max_length": 400},
    },
    {
        "id": "portrait.qwen_image21",
        "title": "Qwen-Image 2.1 character portrait (text to image)",
        "capability": "portrait",
        "workflow": lambda: wf_qwen_image21(False),
        "models": [
            "qwen_image21_int8@draft",
            "qwen_image21_int8@standard",
            "qwen_image21_bf16@max",
            "qwen3vl_8b_int8",
            "qwen_image21_vae",
        ],
        "licence": APACHE,
        "default": True,
        "defaults": {
            "width": 832,
            "height": 1216,
            "steps": 25,
            "cfg": 1.0,
            "cache": "auto",
            "unet": M("qwen_image21_int8"),
        },
        "profiles": {"max": {"unet": M("qwen_image21_bf16"), "steps": 40}},
    },
    {
        "id": "sheet.qwen_image21",
        "title": "Qwen-Image 2.1 character sheet from 1-4 references",
        "capability": "sheet",
        "workflow": lambda: wf_qwen_image21(True),
        "models": [
            "qwen_image21_int8@draft",
            "qwen_image21_int8@standard",
            "qwen_image21_bf16@max",
            "qwen3vl_8b_int8",
            "qwen_image21_vae",
        ],
        "licence": APACHE,
        "default": True,
        "defaults": {
            "width": 2048,
            "height": 1152,
            "resolution": 1024,
            "steps": 25,
            "cfg": 1.0,
            "cache": "auto",
            "unet": M("qwen_image21_int8"),
        },
        "profiles": {"max": {"unet": M("qwen_image21_bf16"), "steps": 40}},
    },
    {
        "id": "keyframe.qwen_edit2511",
        "title": "Qwen-Image-Edit-2511 sample-scene keyframes and turnarounds",
        "capability": "keyframe",
        "workflow": wf_keyframe_qwen_edit,
        "models": [
            "qwen_edit2511_int8",
            "qwen_edit_lightning",
            "qwen_edit_multiangle",
            "qwen25_vl_7b_fp8",
            "qwen_image_vae",
        ],
        "licence": APACHE,
        "default": True,
        "defaults": {"steps": 4, "cfg": 1.0, "lightning_lora": M("qwen_edit_lightning"), "angle_lora": None},
        "profiles": {"max": {"lightning_lora": None, "steps": 40, "cfg": 4.0}},
    },
    {
        "id": "rebuild.wan_animate2",
        "title": "Wan-Animate-2 motion transfer (new scene, same performance)",
        "capability": "rebuild",
        "workflow": wf_wan_animate2,
        "models": [
            "wan_animate2_int8@draft",
            "wan_animate2_int8@standard",
            "wan_animate2_bf16@max",
            "lightx2v_i2v_480p",
            "umt5_xxl_fp8",
            "wan21_vae_bf16",
            "clip_vision_h",
        ],
        "licence": APACHE,
        "default": True,
        "constraints": {"dim_multiple": 16, "window": 81, "overlap": 1, "native_fps": 24},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "lcm",
            "scheduler": "simple",
            "shift": 5.0,
            "speed_lora": M("lightx2v_i2v_480p"),
            "pose_cache": "gpu",
            "unet": M("wan_animate2_int8"),
            "reference_strength": 1.0,
        },
        "profiles": VIDEO_PROFILES_A2,
        "links": ["https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B"],
    },
    {
        "id": "rebuild.scail2_animate",
        "title": "SCAIL-2 animation mode (multi-character, explicit identities)",
        "capability": "rebuild",
        "workflow": lambda: wf_scail2(False),
        "models": [
            "scail2_nvfp4@draft",
            "scail2_nvfp4@standard",
            "scail2_fp16@max",
            "scail2_dpo",
            "lightx2v_i2v_480p",
            "umt5_xxl_fp8",
            "wan21_vae_bf16",
            "clip_vision_h",
            "sam31",
        ],
        "licence": {"weights": "MIT", "code": "Apache-2.0", "commercial": True},
        "constraints": {"dim_multiple": 32, "window": 81, "overlap": 5, "preferred_height": 704, "max_people": 4},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "shift": 5.0,
            "relight_lora": None,
            "dpo_lora": M("scail2_dpo"),
            "speed_lora": M("lightx2v_i2v_480p"),
            "unet": M("scail2_nvfp4"),
            "track_text": "person",
            "max_people": 4,
        },
        "profiles": {"max": {"unet": M("scail2_fp16"), "speed_lora": None, "steps": 40, "cfg": 5.0}},
    },
    {
        "id": "swap.scail2",
        "title": "SCAIL-2 replacement mode (keep the shot, swap 1-4 performers)",
        "capability": "swap",
        "workflow": lambda: wf_scail2(True),
        "models": [
            "scail2_nvfp4@draft",
            "scail2_nvfp4@standard",
            "scail2_fp16@max",
            "scail2_relight",
            "scail2_dpo",
            "lightx2v_i2v_480p",
            "umt5_xxl_fp8",
            "wan21_vae_bf16",
            "clip_vision_h",
            "sam31",
        ],
        "licence": {"weights": "MIT", "code": "Apache-2.0", "commercial": True},
        "default": True,
        "constraints": {"dim_multiple": 32, "window": 81, "overlap": 5, "preferred_height": 704, "max_people": 4},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "shift": 5.0,
            "relight_lora": M("scail2_relight"),
            "dpo_lora": M("scail2_dpo"),
            "speed_lora": M("lightx2v_i2v_480p"),
            "unet": M("scail2_nvfp4"),
            "track_text": "person",
            "max_people": 4,
        },
        "profiles": {"max": {"unet": M("scail2_fp16"), "speed_lora": None, "steps": 40, "cfg": 5.0}},
    },
    {
        "id": "swap.wan22_animate_mix",
        "title": "Wan2.2-Animate Mix mode (swap one performer, relit)",
        "capability": "swap",
        "workflow": wf_wan22_animate_mix,
        "models": [
            "wan22_animate_int8@draft",
            "wan22_animate_int8@standard",
            "wan22_animate_bf16@max",
            "wan22_animate_relight",
            "lightx2v_i2v_480p",
            "umt5_xxl_fp8",
            "wan21_vae",
            "clip_vision_h",
            "sam31",
            "yolov10m",
            "vitpose_l",
        ],
        "licence": {
            "weights": "Apache-2.0",
            "code": "Apache-2.0",
            "commercial": True,
            "notes": "pose preprocessing uses Ultralytics YOLOv10 weights (AGPL-3.0)",
        },
        "node_packs": ["ComfyUI-WanAnimatePreprocess"],
        "constraints": {"dim_multiple": 16, "window": 77, "overlap": 5, "max_people": 1},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "shift": 5.0,
            "relight_lora": M("wan22_animate_relight"),
            "speed_lora": M("lightx2v_i2v_480p"),
            "unet": M("wan22_animate_int8"),
            "performer_index": "0",
            "mask_grow": 16,
        },
        "profiles": {"max": {"unet": M("wan22_animate_bf16"), "speed_lora": None, "steps": 30, "cfg": 3.0}},
    },
    {
        "id": "swap.bernini_rv2v",
        "title": "Bernini-R reference-guided edit (faces, outfits, props)",
        "capability": "swap",
        "workflow": lambda: wf_bernini(True),
        "models": ["bernini_high_int8", "bernini_low_int8", "lightx2v_t2v_v2", "umt5_xxl_fp8", "wan21_vae_bf16"],
        "licence": APACHE,
        "constraints": {"dim_multiple": 16, "window": 121, "overlap": 0},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "split_step": 3,
            "speed_lora_high": M("lightx2v_t2v_v2"),
            "speed_lora_low": M("lightx2v_t2v_v2"),
        },
    },
    {
        "id": "restyle.bernini_v2v",
        "title": "Bernini-R video-to-video restyle",
        "capability": "restyle",
        "workflow": lambda: wf_bernini(False),
        "models": ["bernini_high_int8", "bernini_low_int8", "lightx2v_t2v_v2", "umt5_xxl_fp8", "wan21_vae_bf16"],
        "licence": APACHE,
        "default": True,
        "constraints": {"dim_multiple": 16, "window": 121, "overlap": 0},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "split_step": 3,
            "speed_lora_high": M("lightx2v_t2v_v2"),
            "speed_lora_low": M("lightx2v_t2v_v2"),
        },
    },
    {
        "id": "restyle.vace_fun",
        "title": "Wan2.2-VACE-Fun restyle locked to depth",
        "capability": "restyle",
        "workflow": lambda: wf_vace_fun(False),
        "models": [
            "vace_fun_high_fp8",
            "vace_fun_low_fp8",
            "lightx2v_t2v_high",
            "lightx2v_t2v_low",
            "umt5_xxl_fp8",
            "wan21_vae",
            "da3_small",
        ],
        "licence": APACHE,
        "constraints": {"dim_multiple": 16, "window": 81, "overlap": 0},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "split_step": 3,
            "strength": 1.0,
        },
    },
    {
        "id": "plate.vace_inpaint",
        "title": "Wan2.2-VACE-Fun clean plate (remove the original performer)",
        "capability": "plate",
        "workflow": lambda: wf_vace_fun(True),
        "models": [
            "vace_fun_high_fp8",
            "vace_fun_low_fp8",
            "lightx2v_t2v_high",
            "lightx2v_t2v_low",
            "umt5_xxl_fp8",
            "wan21_vae",
        ],
        "licence": APACHE,
        "default": True,
        "experimental": True,
        "constraints": {"dim_multiple": 16, "window": 81, "overlap": 0},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "split_step": 3,
            "strength": 1.0,
            "prompt": "empty scene, nobody, same location and lighting",
        },
    },
    {
        "id": "lipsync.latentsync16",
        "title": "LatentSync 1.6 mouth re-sync (realistic faces)",
        "capability": "lipsync",
        "workflow": wf_latentsync,
        "models": [],
        "licence": {
            "weights": "OpenRAIL++",
            "code": "Apache-2.0",
            "commercial": True,
            "notes": "OpenRAIL++ use restrictions apply; weights download on first run",
        },
        "node_packs": ["ComfyUI-LatentSyncWrapper"],
        "default": True,
        "constraints": {"min_face_frac": 0.10},
        "defaults": {"lips_expression": 1.5, "steps": 20},
    },
    {
        "id": "lipsync.infinitetalk",
        "title": "InfiniteTalk audio-driven re-render (stylised close-ups)",
        "capability": "lipsync",
        "workflow": wf_infinitetalk,
        "models": [
            "wan21_i2v_480p_fp8",
            "infinitetalk_single",
            "wav2vec2_chinese_base",
            "lightx2v_i2v_480p",
            "umt5_xxl_fp8",
            "wan21_vae_bf16",
            "clip_vision_h",
        ],
        "licence": APACHE,
        "description": "Regenerates the shot from its first frame driven by the vocal: use on close-ups, not dance shots.",
        "constraints": {"dim_multiple": 16, "window": 81, "overlap": 9},
        "defaults": {
            "steps": 6,
            "cfg": 1.0,
            "sampler": "lcm",
            "scheduler": "simple",
            "audio_scale": 1.0,
            "speed_lora": M("lightx2v_i2v_480p"),
            "prompt": "a person singing to camera",
        },
    },
    {
        "id": "upscale.seedvr2",
        "title": "SeedVR2 video restoration and upscale",
        "capability": "upscale",
        "workflow": wf_seedvr2,
        "models": ["seedvr2_3b_fp8@draft", "seedvr2_7b_nvfp4@standard", "seedvr2_7b_fp8@max", "seedvr2_vae"],
        "licence": APACHE,
        "default": True,
        "defaults": {"unet": M("seedvr2_7b_nvfp4"), "color_correction": "lab"},
        "profiles": {"draft": {"unet": M("seedvr2_3b_fp8")}, "max": {"unet": M("seedvr2_7b_fp8")}},
    },
    {
        "id": "interp.rife",
        "title": "RIFE 4.26 frame interpolation",
        "capability": "interp",
        "workflow": lambda: wf_interp("rife_v426"),
        "models": ["rife_v426"],
        "licence": MIT,
        "default": True,
    },
    {
        "id": "interp.film",
        "title": "FILM frame interpolation",
        "capability": "interp",
        "workflow": lambda: wf_interp("film_net"),
        "models": ["film_net"],
        "licence": APACHE,
    },
    # ---- Docker tool engines --------------------------------------------------------------------
    {
        "id": "separate.roformer",
        "title": "RoFormer stem separation (vocals, lead/backing, dereverb)",
        "capability": "separate",
        "backend": "docker",
        "image": "tools",
        "task": "separate",
        "models": [],
        "licence": {
            "weights": "MIT / GPL-3.0 (dereverb)",
            "code": "MIT",
            "commercial": True,
            "notes": "Kim FT2 and becruily models declare no licence; anvuew dereverb is GPL-3.0",
        },
        "default": True,
        "defaults": {
            "vocals_model": "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
            "karaoke_model": "mel_band_roformer_karaoke_becruily.ckpt",
            "dereverb_model": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        },
    },
    {
        "id": "transcribe.soulx_pre",
        "title": "SoulX-Singer preprocess: F0, notes and word-timed lyrics",
        "capability": "transcribe",
        "backend": "docker",
        "image": "soulx",
        "task": "soulx",
        "default": True,
        "models": ["soulx_preprocess"],
        "licence": APACHE,
        "defaults": {"action": "preprocess"},
    },
    {
        "id": "svs.soulx_singer",
        "title": "SoulX-Singer score-mode singing synthesis (EN, ZH, YUE)",
        "capability": "svs",
        "backend": "docker",
        "image": "soulx",
        "task": "soulx",
        "default": True,
        "models": ["soulx_singer", "soulx_preprocess"],
        "licence": APACHE,
        "defaults": {"action": "sing", "control": "score", "auto_shift": True},
    },
    {
        "id": "svs.acestep_repaint",
        "title": "ACE-Step 1.5 repaint with new lyrics (JA, KO and others)",
        "capability": "svs",
        "backend": "docker",
        "image": "acestep",
        "task": "acestep",
        "experimental": True,
        "models": [],
        "licence": MIT,
        "defaults": {"action": "repaint"},
    },
    {
        "id": "svc.soulx_svc",
        "title": "SoulX-Singer-SVC zero-shot singing voice conversion",
        "capability": "svc",
        "backend": "docker",
        "image": "soulx",
        "task": "soulx",
        "default": True,
        "models": ["soulx_singer", "soulx_preprocess"],
        "licence": APACHE,
        "defaults": {"action": "svc"},
    },
    {
        "id": "svc.applio_rvc",
        "title": "Applio RVC conversion with a trained voice",
        "capability": "svc",
        "backend": "docker",
        "image": "applio",
        "task": "applio",
        "models": [],
        "licence": MIT,
        "defaults": {"action": "infer", "f0_method": "rmvpe", "index_rate": 0.5, "protect": 0.4},
    },
    {
        "id": "voice_train.applio",
        "title": "Applio RVC training (10+ minutes of clean singing)",
        "capability": "voice_train",
        "backend": "docker",
        "image": "applio",
        "task": "applio",
        "default": True,
        "models": [],
        "licence": MIT,
        "defaults": {"action": "train", "sample_rate": 48000, "total_epoch": 300, "batch_size": 8},
    },
    {
        "id": "voice.voxcpm2_design",
        "title": "VoxCPM2 voice design (an invented voice from a description)",
        "capability": "voice",
        "backend": "docker",
        "image": "voxcpm",
        "task": "voxcpm",
        "default": True,
        "models": ["voxcpm2"],
        "licence": APACHE,
        "defaults": {"action": "design", "cfg_value": 2.0, "steps": 10},
    },
    {
        "id": "mix.pedalboard_matchering",
        "title": "Mix and master: pedalboard, pyloudnorm, Matchering",
        "capability": "mix",
        "backend": "docker",
        "image": "tools",
        "task": "mix",
        "gpu": False,
        "default": True,
        "models": [],
        "licence": {"weights": "n/a", "code": "GPL-3.0 (out of process)", "commercial": True},
    },
]


def write(engine: dict[str, Any]) -> None:
    yaml = YAML()
    yaml.width = 120
    spec = {k: v for k, v in engine.items() if k != "workflow"}
    spec.setdefault("backend", "comfyui")
    if spec["backend"] == "comfyui":
        workflow, bindings = engine["workflow"]()
        folder = REPO / "workflows" / engine["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "standard.api.json").write_text(
            json.dumps(workflow, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        with (folder / "bindings.yaml").open("w", encoding="utf-8", newline="\n") as fh:
            yaml.dump(bindings, fh)
        spec["workflow"] = f"{engine['id']}/standard.api.json"
        spec["bindings"] = f"{engine['id']}/bindings.yaml"
    order = [
        "id",
        "title",
        "capability",
        "backend",
        "description",
        "default",
        "experimental",
        "workflow",
        "bindings",
        "image",
        "task",
        "gpu",
        "licence",
        "constraints",
        "models",
        "node_packs",
        "defaults",
        "profiles",
        "links",
    ]
    ordered = {k: spec[k] for k in order if k in spec}
    with (REPO / "engines" / f"{engine['id']}.yaml").open("w", encoding="utf-8", newline="\n") as fh:
        yaml.dump(ordered, fh)


def main() -> None:
    for old in (REPO / "engines").glob("*.yaml"):
        old.unlink()
    for old in (REPO / "workflows").iterdir():
        if old.is_dir():
            shutil.rmtree(old)
    for engine in ENGINES:
        write(engine)
    print(f"wrote {len(ENGINES)} engines")


if __name__ == "__main__":
    main()
