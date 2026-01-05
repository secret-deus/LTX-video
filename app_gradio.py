import os
import subprocess
import uuid
from pathlib import Path
import gradio as gr

MODEL_TO_CFG = {
    "ltxv-13b-0.9.8-distilled": "configs/ltxv-13b-0.9.8-distilled.yaml",
    "ltxv-13b-0.9.8-distilled-fp8": "configs/ltxv-13b-0.9.8-distilled-fp8.yaml",
    "ltxv-2b-0.9.8-distilled-fp8": "configs/ltxv-2b-0.9.8-distilled-fp8.yaml",
}

DEFAULT_MODEL = os.environ.get("LTX_MODEL", "ltxv-2b-0.9.8-distilled-fp8")
OFFLOAD = os.environ.get("LTX_OFFLOAD", "0") == "1"

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True, parents=True)

def run_inference(model_name, prompt, neg_prompt, width, height, num_frames, fps, seed):
    cfg = MODEL_TO_CFG.get(model_name)
    if not cfg:
        raise ValueError(f"Unsupported model: {model_name}")

    job_id = str(uuid.uuid4())[:8]
    out_folder = OUTDIR / f"{model_name}_{job_id}"
    out_folder.mkdir(exist_ok=True, parents=True)

    # ✅ 注意：使用官方 inference.py 的参数名
    cmd = [
        "python3", "inference.py",
        "--pipeline_config", cfg,
        "--prompt", prompt,
        "--negative_prompt", neg_prompt or "",
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--num_frames", str(int(num_frames)),
        "--frame_rate", str(int(fps)),
        "--seed", str(int(seed)),
        "--output_path", str(out_folder),
    ]

    if OFFLOAD:
        cmd += ["--offload_to_cpu", "True"]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        raise RuntimeError(f"Inference failed:\n{logs}")

    # 推断输出文件：取目录里最新的 mp4
    mp4s = sorted(out_folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        # 有些版本输出可能是 gif / mov / webm，兜底
        vids = sorted(list(out_folder.glob("*.*")), key=lambda p: p.stat().st_mtime, reverse=True)
        if vids:
            return str(vids[0]), logs
        raise RuntimeError(f"Generation succeeded but no video found in {out_folder}\n{logs}")

    return str(mp4s[0]), logs

with gr.Blocks() as demo:
    gr.Markdown("# LTX-Video WebUI (Docker Compose)\nSupports 13B distilled / 13B distilled FP8 / 2B distilled FP8")

    model_name = gr.Dropdown(choices=list(MODEL_TO_CFG.keys()), value=DEFAULT_MODEL, label="Model")
    prompt = gr.Textbox(label="Prompt", value="A cinematic shot of a cat walking in the rain")
    neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality")

    with gr.Row():
        width = gr.Slider(256, 1536, step=32, value=640, label="Width")
        height = gr.Slider(256, 1536, step=32, value=384, label="Height")

    with gr.Row():
        num_frames = gr.Slider(16, 257, step=1, value=49, label="Frames")
        fps = gr.Slider(8, 60, step=1, value=24, label="FPS")

    with gr.Row():
        seed = gr.Number(value=42, label="Seed")

    btn = gr.Button("Generate")
    video = gr.Video(label="Result Video")
    logs = gr.Textbox(label="Logs", lines=12)

    btn.click(
        fn=run_inference,
        inputs=[model_name, prompt, neg_prompt, width, height, num_frames, fps, seed],
        outputs=[video, logs]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)