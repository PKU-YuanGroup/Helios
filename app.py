import tempfile
import time

import gradio as gr
import spaces
import torch

from torch.utils._pytree import tree_map
from diffusers import AutoencoderKLWan, HeliosDMDScheduler, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_image, load_video


# ---------------------------------------------------------------------------
# Pre-load model
# ---------------------------------------------------------------------------
MODEL_ID = "BestWishYsh/Helios-Distilled"

vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
scheduler = HeliosDMDScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
pipe = HeliosPyramidPipeline.from_pretrained(
    MODEL_ID, vae=vae, scheduler=scheduler, torch_dtype=torch.bfloat16, is_distilled=True
)
pipe.to("cuda")

cuda_major = torch.cuda.get_device_capability()[0]
if cuda_major >= 9:
    # H100/H800 (SM90+) with FA3
    try:
        pipe.transformer.set_attention_backend("_flash_3_hub")
    except Exception:
        pipe.transformer.set_attention_backend("flash_hub")
else:
    # 4090/A100 etc (SM89+) with FA2
    pipe.transformer.set_attention_backend("flash_hub")

# ---------------------------------------------------------------------------
# AoTI
# ---------------------------------------------------------------------------

# Dynamic shapes: within a generation, only hidden_states H/W change across
# pyramid stages (history latents stay at full resolution). text_seq_length
# varies between different prompts.
_AUTO = torch.export.Dim.AUTO

TRANSFORMER_DYNAMIC_SHAPES = {
    "hidden_states": {
        3: _AUTO,  # H — doubles each pyramid stage
        4: _AUTO,  # W — doubles each pyramid stage
    },
    "encoder_hidden_states": {
        1: _AUTO,  # text_seq_length — varies with prompt
    },
}

INDUCTOR_CONFIGS = {
    "conv_1x1_as_mm": True,
    "epilogue_fusion": False,
    "coordinate_descent_tuning": True,
    "coordinate_descent_check_all_directions": True,
    # "max_autotune": True,
    "triton.cudagraphs": True,
}

@spaces.GPU(duration=1500) # maximum duration allowed during startup
def compile_transformer():
    with spaces.aoti_capture(pipe.transformer) as call:
        pipe(
            "arbitrary example prompt",
            height=384,
            width=640,
            num_frames=33,
            guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
            pyramid_num_inference_steps_list=[2, 2, 2],
            is_amplify_first_chunk=True,
        )

    dynamic_shapes = tree_map(lambda t: None, call.kwargs)
    dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

    with torch.no_grad():
        exported = torch.export.export(
            pipe.transformer,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )

    return spaces.aoti_compile(exported, INDUCTOR_CONFIGS)

compiled_transformer = compile_transformer()
spaces.aoti_apply(compiled_transformer, pipe.transformer)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@spaces.GPU(duration=60)
def generate_video(
    mode: str,
    prompt: str,
    image_input,
    video_input,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    is_amplify_first_chunk: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt:
        raise gr.Error("Please provide a prompt.")

    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    kwargs = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "guidance_scale": 1.0,
        "generator": generator,
        "output_type": "np",
        "pyramid_num_inference_steps_list": [
            int(num_inference_steps),
            int(num_inference_steps),
            int(num_inference_steps),
        ],
        "is_amplify_first_chunk": is_amplify_first_chunk,
    }

    if mode == "Image-to-Video" and image_input is not None:
        img = load_image(image_input).resize((int(width), int(height)))
        kwargs["image"] = img
    elif mode == "Video-to-Video" and video_input is not None:
        kwargs["video"] = load_video(video_input)

    t0 = time.time()
    output = pipe(**kwargs).frames[0]
    elapsed = time.time() - t0

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    export_to_video(output, tmp.name, fps=24)
    info = f"Generated in {elapsed:.1f}s · {num_frames} frames · {height}×{width}"
    return tmp.name, info


# ---------------------------------------------------------------------------
# UI Setup
# ---------------------------------------------------------------------------
def update_conditional_visibility(mode):
    if mode == "Image-to-Video":
        return gr.update(visible=True), gr.update(visible=False)
    elif mode == "Video-to-Video":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


CSS = """
#header { text-align: center; margin-bottom: 1.5em; }
#header h1 { font-size: 2.2em; margin-bottom: 0.2em; }
.logo { max-height: 100px; margin: 0 auto 10px auto; display: block; }
.link-buttons { display: flex; justify-content: center; gap: 15px; margin-top: 10px; }
.link-buttons a {
    background-color: #2b3137;
    color: #ffffff !important;
    padding: 8px 20px;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
    font-size: 1em;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.link-buttons a:hover { background-color: #4a535c; transform: translateY(-1px); }
.contain { max-width: 1350px; margin: 0 auto !important; }
"""

with gr.Blocks(title="Helios Video Generation") as demo:
    gr.HTML(
        """
        <div style='display: flex; align-items: center; justify-content: center; width: 100%;'>
            <img src="https://raw.githubusercontent.com/SHYuanBest/shyuanbest_media/main/Helios/logo_white.png" style='width: 400px; height: auto;' />
        </div>
        <div id="header">
            <h1>🎬 Helios 14B Distilled: Real Real-Time Long Video Generation Model</h1>
            <p style="font-size: 1.1em; color: #666; margin-top: 0.5em; margin-bottom: 1em;">
                If you like our project, please give us a star ⭐ on GitHub for the latest update.
            </p>
            <div class="link-buttons">
                <a href="https://github.com/PKU-YuanGroup/Helios" target="_blank">💻 Code</a>
                <a href="https://pku-yuangroup.github.io/Helios-Page" target="_blank">📄 Page</a>
                <a href="https://www.youtube.com/watch?v=vd_AgHtOUFQ" target="_blank">🎥 Main Feature</a>
                <a href="https://www.youtube.com/watch?v=1GeIU2Dn7UY" target="_blank">⚡ Inference Speed</a>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["Text-to-Video", "Image-to-Video", "Video-to-Video"],
                value="Text-to-Video",
                label="Generation Mode",
            )
            image_input = gr.Image(label="Image (for I2V)", type="filepath", visible=False)
            video_input = gr.Video(label="Video (for V2V)", visible=False)
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                value=(
                    "A vibrant tropical fish swimming gracefully among colorful coral reefs in "
                    "a clear, turquoise ocean. The fish has bright blue and yellow scales with a "
                    "small, distinctive orange spot on its side, its fins moving fluidly. The coral "
                    "reefs are alive with a variety of marine life, including small schools of "
                    "colorful fish and sea turtles gliding by. The water is crystal clear, allowing "
                    "for a view of the sandy ocean floor below. The reef itself is adorned with a mix "
                    "of hard and soft corals in shades of red, orange, and green. The photo captures "
                    "the fish from a slightly elevated angle, emphasizing its lively movements and the "
                    "vivid colors of its surroundings. A close-up shot with dynamic movement."
                ),
            )
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    height = gr.Number(value=384, label="Height", precision=0, interactive=False)
                    width = gr.Number(value=640, label="Width", precision=0, interactive=False)
                with gr.Row():
                    num_frames = gr.Slider(33, 231, value=231, step=33, label="Num Frames")
                    num_inference_steps = gr.Slider(1, 10, value=2, step=1, label="Steps per stage")
                with gr.Row():
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    is_amplify_first_chunk = gr.Checkbox(label="Amplify First Chunk", value=True)

            generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_output = gr.Video(label="Generated Video", autoplay=True)
            info_output = gr.Textbox(label="Info", interactive=False)

    mode.change(fn=update_conditional_visibility, inputs=[mode], outputs=[image_input, video_input])
    generate_btn.click(
        fn=generate_video,
        inputs=[
            mode,
            prompt,
            image_input,
            video_input,
            height,
            width,
            num_frames,
            num_inference_steps,
            seed,
            is_amplify_first_chunk,
        ],
        outputs=[video_output, info_output],
    )

    gr.Examples(
        examples=[
            [
                "Text-to-Video",
                "A vibrant tropical fish swimming gracefully among colorful coral reefs in "
                "a clear, turquoise ocean. The fish has bright blue and yellow scales with a "
                "small, distinctive orange spot on its side, its fins moving fluidly. The coral "
                "reefs are alive with a variety of marine life, including small schools of "
                "colorful fish and sea turtles gliding by. The water is crystal clear, allowing "
                "for a view of the sandy ocean floor below. The reef itself is adorned with a mix "
                "of hard and soft corals in shades of red, orange, and green. The photo captures "
                "the fish from a slightly elevated angle, emphasizing its lively movements and the "
                "vivid colors of its surroundings. A close-up shot with dynamic movement.",
                None,
                None,
            ],
            [
                "Text-to-Video",
                "An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in "
                "thought pondering the history of the universe as he sits at a cafe in Paris, his eyes "
                "focus on people offscreen as they walk as he sits mostly motionless, he is dressed in "
                "a wool coat suit coat with a button-down shirt , he wears a brown beret and glasses "
                "and has a very professorial appearance, and the end he offers a subtle closed-mouth "
                "smile as if he found the answer to the mystery of life, the lighting is very cinematic "
                "with the golden light and the Parisian streets and city in the background, depth of "
                "field, cinematic 35mm film.",
                None,
                None,
            ],
            [
                "Text-to-Video",
                "A drone camera circles around a beautiful historic church built on a rocky outcropping "
                "along the Amalfi Coast, the view showcases historic and magnificent architectural "
                "details and tiered pathways and patios, waves are seen crashing against the rocks "
                "below as the view overlooks the horizon of the coastal waters and hilly landscapes "
                "of the Amalfi Coast Italy, several distant people are seen walking and enjoying vistas "
                "on patios of the dramatic ocean views, the warm glow of the afternoon sun creates a "
                "magical and romantic feeling to the scene, the view is stunning captured with beautiful photography.",
                None,
                None,
            ],
            [
                "Image-to-Video",
                "A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and respect for nature’s might.",
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg",
                None,
            ],
            [
                "Video-to-Video",
                "A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery.",
                None,
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4",
            ],
        ],
        inputs=[mode, prompt, image_input, video_input],
        label="Example Prompts",
    )

if __name__ == "__main__":
    demo.launch(share=True, css=CSS, theme=gr.themes.Soft())
