import os
import subprocess
import uuid
from pathlib import Path
import gradio as gr

# 只支持你指定的三个模型
MODEL_TO_CFG = {
    "ltxv-13b-0.9.8-distilled": "configs/ltxv-13b-0.9.8-distilled.yaml",
    "ltxv-13b-0.9.8-distilled-fp8": "configs/ltxv-13b-0.9.8-distilled-fp8.yaml",
    "ltxv-2b-0.9.8-distilled-fp8": "configs/ltxv-2b-0.9.8-distilled-fp8.yaml",
}

DEFAULT_MODEL = os.environ.get("LTX_MODEL", "ltxv-13b-0.9.8-distilled")
OFFLOAD = os.environ.get("LTX_OFFLOAD", "0") == "1"

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True, parents=True)

def run_inference(model_name, prompt, neg_prompt, width, height, num_frames, fps, seed, steps):
    cfg = MODEL_TO_CFG.get(model_name)
    if not cfg:
        raise ValueError(f"Unsupported model: {model_name}")

    # 输出文件路径
    job_id = str(uuid.uuid4())[:8]
    out_path = OUTDIR / f"{model_name}_{job_id}.mp4"

    # inference.py 是官方推理入口（README 明确）
    cmd = [
        "python3", "inference.py",
        "--config", cfg,
        "--prompt", prompt,
        "--negative_prompt", neg_prompt or "",
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--num_frames", str(int(num_frames)),
        "--fps", str(int(fps)),
        "--seed", str(int(seed)),
        "--steps", str(int(steps)),
        "--output_path", str(out_path),
    ]

    # 如果启用 offload（官方支持 offloading unused parts to CPU）
    if OFFLOAD:
        cmd += ["--offload_to_cpu"]

    # 执行并捕获日志
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        raise RuntimeError(f"Inference failed:\n{logs}")

    return str(out_path), logs


with gr.Blocks() as demo:
    gr.Markdown("# LTX-Video WebUI (Docker Compose)\nOnly supports: 13B distilled / 13B distilled FP8 / 2B distilled FP8")

    with gr.Row():
        model_name = gr.Dropdown(
            choices=list(MODEL_TO_CFG.keys()),
            value=DEFAULT_MODEL,
            label="Model"
        )

    prompt = gr.Textbox(label="Prompt", value="A cinematic shot of a cat walking in the rain, ultra realistic.")
    neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality")

    with gr.Row():
        width = gr.Slider(256, 1536, step=32, value=768, label="Width")
        height = gr.Slider(256, 1536, step=32, value=512, label="Height")

    with gr.Row():
        num_frames = gr.Slider(16, 1024, step=1, value=97, label="Frames")
        fps = gr.Slider(8, 60, step=1, value=24, label="FPS")

    with gr.Row():
        seed = gr.Number(value=42, label="Seed")
        steps = gr.Slider(1, 80, step=1, value=30, label="Steps")

    btn = gr.Button("Generate")
    video = gr.Video(label="Result Video")
    logs = gr.Textbox(label="Logs", lines=12)

    btn.click(
        fn=run_inference,
        inputs=[model_name, prompt, neg_prompt, width, height, num_frames, fps, seed, steps],
        outputs=[video, logs]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)