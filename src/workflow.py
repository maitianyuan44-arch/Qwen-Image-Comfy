"""ComfyUI node graph for Qwen-Image 2.1, built in code per job.

The graph is a faithful transcription of ComfyUI's official templates
(Comfy-Org/workflow_templates: image_qwen_image_2_1_t2i.json and
image_qwen_image_2_1_image_edit.json), minus the optional prompt-enhancer
branch (a second ~9.5 GB text model behind a "Generate Text" node, off by
default in the official template and not wired here):

    UNETLoader ──> QwenImage21Cache ──> KSampler.model
    CLIPLoader ──> TextEncodeQwenImage21 ──> KSampler.positive / .negative
    EmptyLatentImage ──────────────────────> KSampler.latent_image   (t2i)
    VAELoader ──> VAEDecode.vae
    KSampler ──> VAEDecode ──> SaveImage

Edit mode (a job with an input image) reuses the same graph with two
changes, exactly as the official edit template does: the image is fed to
TextEncodeQwenImage21 as `images.image_1` together with the VAE (so the
encoder splices the reference's VAE latents into the sequence), and
KSampler's latent_image comes from the encode node's own `latent` output
— an empty latent on the (resized) reference's size — instead of
EmptyLatentImage. Sampling runs at denoise 1.0: this is the model's
instruction-edit path, not a classic img2img blend.

Node schemas verified against ComfyUI v0.46.2 source
(comfy_extras/nodes_qwen.py): TextEncodeQwenImage21 inputs are clip /
prompt / negative_prompt / vae? / resolution / images.image_N (autogrow);
QwenImage21Cache inputs are model / device / dtype. These nodes do not
exist in the ComfyUI 0.36.0 the H3 worker pins — hence this worker's own
newer pin in the Dockerfile.
"""

# Model files, as laid out in Comfy-Org/Qwen-Image-2.1 (the Model Cache
# snapshot is symlinked into these exact category dirs by model_link.py).
UNET_NAME = "qwen_image_2.1_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_8b_int8_convrot.safetensors"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"

OUTPUT_NODE_ID = "save_image"


def build_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 25,
    cfg: float = 1.0,
    image_name: str | None = None,
    edit_resolution: int = 1024,
    filename_prefix: str = "qwen_image/job",
) -> dict:
    """Returns the API-format graph: {node_id: {"class_type", "inputs"}}.

    image_name: a file already uploaded to ComfyUI's input/ directory.
    When given, the job is an edit of that image; otherwise text-to-image
    at width x height.
    """
    nodes: dict[str, dict] = {
        "unet_loader": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        # KV-cache placement for the 2.1 transformer; the official
        # templates keep the defaults ('auto' device, lossless dtype).
        "model_cache": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["unet_loader", 0], "device": "auto", "dtype": "default"},
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image", "device": "default"},
        },
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
    }

    encode_inputs: dict = {
        "clip": ["clip_loader", 0],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        # TextEncodeQwenImage21's resolution input is required in API
        # format even for text-to-image (validation rejects the graph
        # without it). It only resizes reference images, so with no
        # references it is unused — but it must be present.
        "resolution": edit_resolution,
    }
    if image_name is not None:
        nodes["load_image"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }
        encode_inputs["vae"] = ["vae_loader", 0]
        encode_inputs["images.image_1"] = ["load_image", 0]
        latent_ref = ["encode", 2]
    else:
        nodes["empty_latent"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
        latent_ref = ["empty_latent", 0]
    nodes["encode"] = {"class_type": "TextEncodeQwenImage21", "inputs": encode_inputs}

    nodes["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["model_cache", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["encode", 0],
            "negative": ["encode", 1],
            "latent_image": latent_ref,
            "denoise": 1.0,
        },
    }
    nodes["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae_loader", 0]},
    }
    nodes[OUTPUT_NODE_ID] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix},
    }
    return nodes
