# Qwen-Image Worker (worker-qwen)

RunPod Serverless worker for **Qwen-Image 2.1** image generation in the
RunPod Media Studio app, built in the same fkky pattern as the MiniMax H3
worker (`worker-lora/`): a GitHub-deployed image that installs ComfyUI,
links the endpoint's **Model Cache** snapshot into ComfyUI's `models/`
tree at boot (no network volume, no monthly fee), builds the ComfyUI
graph in code from simple fields, and uploads the result PNG to the S3
bucket, returning a presigned URL.

```
worker-qwen/
├── Dockerfile          ComfyUI 0.46.2 + torch cu130, no models baked in
├── requirements.txt    runpod / requests / websocket-client / boto3
└── src/
    ├── handler.py      RunPod entry point; boots ComfyUI as a subprocess
    ├── config.py       Env-driven config (HF_REPO, S3_*, defaults)
    ├── validator.py    App-shaped input -> cleaned params
    ├── workflow.py     The Qwen-Image 2.1 graph, built per job
    ├── comfy_client.py ComfyUI HTTP/WebSocket client
    ├── model_link.py   Model Cache snapshot -> ComfyUI models/ symlinks
    ├── image.py        Edit-target download/decode
    ├── storage.py      S3 upload + presigned URL
    ├── security.py     SSRF guard for image_url
    ├── errors.py       Shared error contract with the H3 worker
    └── watchdog.py     Pod-mode self-termination (no-op on Serverless)
```

## App contract — no app changes needed

The app (RunPodClient.kt) sends, for its Qwen-Image model:

```json
{"input": {
  "workflow": { "...": "placeholder asset, ignored by this worker" },
  "prompt": "...", "negative_prompt": "...",
  "image_base64": "data:image/jpeg;base64,...",   // only when a photo is picked
  "width": 1024, "height": 576                     // only with the photo (its aspect)
}}
```

This worker reads the flat fields, ignores `workflow`, and builds the
real graph itself. With an image it runs the model's **edit** graph (the
photo becomes the encode node's reference and sets the canvas); without
one it runs text-to-image (default 1024×1024 when no size is sent).
It returns:

```json
{"output": {"status": "completed", "image_url": "https://...presigned...",
            "meta": {"mode": "t2i|edit", "seed": 0, "steps": 25, "...": "..."},
            "timing": {"inference_s": 0.0, "...": "..."}}}
```

`image_url` is one of the keys the app's output parser looks for, so the
existing app build works against this endpoint unchanged.

## Model files (from Comfy-Org/Qwen-Image-2.1)

Only three files are loaded by the graph — the INT8 set:

| Folder | File | Size |
|---|---|---|
| `diffusion_models/` | `qwen_image_2.1_int8_convrot.safetensors` | 7.26 GB |
| `text_encoders/` | `qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| `vae/` | `qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |
| | **Total** | **17.28 GB** |

Not needed, deliberately excluded:

- `qwen3.5_9b_qwen_image_2.1_pe_*` (9.47 GB each) — only used by the
  optional prompt-enhancer ("Generate Text") branch of the official
  template, which is off by default there and not wired here.
- The bf16 / turbo / LoRA / ControlNet variants in the same repo.

Graph (transcribed from the official templates
`image_qwen_image_2_1_t2i.json` / `image_qwen_image_2_1_image_edit.json`):
UNETLoader → QwenImage21Cache → KSampler (euler / simple, 25 steps,
cfg 1, denoise 1); CLIPLoader (`type=qwen_image`) → TextEncodeQwenImage21
→ KSampler positive/negative; EmptyLatentImage (t2i) or the encode node's
own latent (edit); VAEDecode → SaveImage.

## Cached model field — read this first

The endpoint's **Cached model** field takes ONE Hugging Face repo, and
Model Cache downloads that whole repo before the first deploy finishes.

- **Recommended: a slim fork repo (~17 GB).** Create a new model repo on
  Hugging Face and put ONLY the three files above in it, keeping the
  folder names (`diffusion_models/…`, `text_encoders/…`, `vae/…`) — the
  same trick as `fkyyy/minimax-h3-comfy-slim`. One-time upload of ~17 GB
  (the Hugging Face website's upload, or `huggingface-cli upload`).
  Type the fork's id (lowercase) into the Cached model field and set the
  endpoint env `HF_REPO` to the same lowercase id.
- **Fallback: `Comfy-Org/Qwen-Image-2.1` directly.** No upload needed and
  the layout matches, but the repo totals **122.8 GB** (all variants), so
  the first cache download is huge — a 75 GB repo was already seen
  stalling in Model Cache's "pending download" state on the H3 project.
  Use only if the slim fork is not an option. `HF_REPO` then stays at its
  default (`comfy-org/qwen-image-2.1`).

## Endpoint settings

| Setting | Value |
|---|---|
| Type | Queue based |
| GPU | **24 GB × 1** (weights are 17.3 GB; at 1024×1024 the two int8 models plus VAE and activations fit, and ComfyUI can offload the text encoder to RAM if pressed) |
| Max workers / Active workers | 1 / 0 ($0 while idle) |
| Initialization timeout | 420 s |
| Execution timeout | 1800 s (enable it and set the value) |
| Container disk | 20 GB |
| Cached model | slim fork id (see above) |
| Network volumes | none |
| Data centers | AP-JP-1, AP-IN-1, CA-MTL-1, CA-MTL-3, CA-MTL-4 only (set after creation: Manage → Edit endpoint → Advanced) |
| FlashBoot | leave OFF — the free tier's 3 slots are already used by the other endpoints |

Environment variables (Raw editor; the five S3 lines are the same values
as the H3 endpoint's — copy them from there):

```
S3_ENDPOINT=...
S3_REGION=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET_DEFAULT=...
HF_REPO=<cached-model repo id, lowercase>   # only when using a slim fork
```

Optional tuning (defaults in `src/config.py`): `DEFAULT_STEPS=25`,
`DEFAULT_CFG_SCALE=1.0`, `DEFAULT_WIDTH`/`DEFAULT_HEIGHT=1024`,
`COMFY_INFERENCE_TIMEOUT_S=900`.

## Setup steps (RunPod console)

Same flow as the H3 / R2V endpoints:

1. **Prepare the model repo** (once): create the slim fork on Hugging
   Face with the three files in their folders (see above).
2. **Publish this worker**: put the *contents* of this `worker-qwen/`
   folder at the root of a new GitHub repository (Dockerfile and `src/`
   at the top level).
3. RunPod → **Serverless → New Endpoint → Deploy from a GitHub
   repository** → select the repo, branch `main`, Dockerfile path
   `/Dockerfile` → **Queue** (not Load balancer). If the "could not find
   runpod.serverless.start()" warning appears, choose **Continue anyway**
   (same as the H3 endpoints; the call is at module level in
   `src/handler.py`).
4. Configure: name e.g. `Qwen-Image-Comfy`; GPU **24 GB** × 1;
   Initialization timeout **420**; enable Execution timeout and set
   **1800**; **Cached model** = the repo id from step 1; Container disk
   **20**; Max workers **1**, Active workers **0**; FlashBoot off.
5. **Deploy**, then **Manage → Edit endpoint → Advanced → Data centers**:
   keep only AP-JP-1, AP-IN-1, CA-MTL-1, CA-MTL-3, CA-MTL-4; confirm no
   network volume is attached.
6. **Edit endpoint → Environment variables** (Raw editor): paste the five
   S3 lines (copied from the H3 endpoint) plus `HF_REPO` if using a slim
   fork → **Update Variables**.
7. Copy the new **Endpoint ID** into the app: 設定 tab → "Qwen-Image の
   Endpoint ID" (replacing the old dead one).
8. First test in the app: model **Qwen-Image**, a short prompt, no
   image. The first job takes minutes (17 GB of weights load); after
   that, a 1024×1024 image should take well under a minute of GPU time.

## Notes / risks

- **ComfyUI is pinned to 0.46.2 here, not the H3 worker's 0.36.0** —
  the Qwen-Image 2.1 nodes (`TextEncodeQwenImage21`, `QwenImage21Cache`)
  don't exist in 0.36.0 (verified against ComfyUI tags; present since
  ≥0.38.0). The two workers are separate repos/images, so the pins don't
  conflict. Node schemas in `src/workflow.py` were verified against
  v0.46.2's `comfy_extras/nodes_qwen.py`.
- The image needs a CUDA 13-capable host driver (the torch cu130 swap,
  same as the H3 image). If the console offers a minimum CUDA version /
  driver filter, prefer the newest available.
- No batch mode (the H3 worker has one; the app never sends batch jobs
  for images, so it was left out).
- Edit mode follows the official edit template: the app's composed
  descriptive prompt is passed to the edit graph as-is. It behaves as
  "regenerate this picture as described", not as a pixel-preserving
  img2img blend (denoise is fixed at 1.0, as in the template).

ak
