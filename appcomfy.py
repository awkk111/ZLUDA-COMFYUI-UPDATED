import gradio as gr
import torch
import numpy as np
import random
import gc
import os
from PIL import Image
import imageio
from base64 import b64encode
import torch_directml  # For DirectML support
# Add ZLUDA support if it's available
# from zcuda import init, set_device  # Uncomment if ZLUDA is available

# Assuming necessary imports for ZLUDA are included.
# If ZLUDA is available, we would use something like this:
# from zcuda import init, set_device


# Define your model loading functions here (similar to your original code)
# Initialize nodes (same as before)
# Assuming your model initialization functions are imported correctly.
from comfy import model_management
from nodes import (
    CheckpointLoaderSimple, CLIPLoader, CLIPTextEncode, VAEDecode, VAELoader,
    KSampler, UNETLoader, LoadImage, CLIPVisionLoader, CLIPVisionEncode
)
from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_images import SaveAnimatedWEBP
from comfy_extras.nodes_video import SaveWEBM
from comfy_extras.nodes_wan import WanImageToVideo


# Function to clear memory
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    for obj in list(globals().values()):
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            del obj
    gc.collect()

def set_device(device_type):
    """Sets the device based on the input device type"""
    if device_type == "CPU":
        print("Using CPU")
        device = torch.device("cpu")
    elif device_type == "DirectML":
        print("Using DirectML")
        device = torch.device("cuda", device=0) if torch.cuda.is_available() else torch.device("cpu")
        device = torch_directml.device()  # Assume DirectML is used if available
    elif device_type == "ZLUDA":
        print("Using ZLUDA")
        # Implement ZLUDA initialization here (assuming the ZLUDA setup is correct)
        # set_device(device_index)  # Uncomment and use actual ZLUDA device setup
        device = torch.device("cuda")  # Placeholder for ZLUDA-compatible hardware
    else:
        raise ValueError("Unsupported device type")
    
    torch.set_default_device(device)
    return device

def save_as_mp4(images, filename_prefix, fps, output_dir="/content/ComfyUI/output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"

    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_webp(images, filename_prefix, fps, quality=90, lossless=False, method=4, output_dir="/content/ComfyUI/output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webp"

    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    kwargs = {
        'fps': int(fps),
        'quality': int(quality),
        'lossless': bool(lossless),
        'method': int(method)
    }

    with imageio.get_writer(output_path, format='WEBP', mode='I', **kwargs) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_webm(images, filename_prefix, fps, codec="vp9", quality=32, output_dir="/content/ComfyUI/output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webm"

    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    kwargs = {
        'fps': int(fps),
        'quality': int(quality),
        'codec': str(codec),
        'output_params': ['-crf', str(int(quality))]
    }

    with imageio.get_writer(output_path, format='FFMPEG', mode='I', **kwargs) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_image(image, filename_prefix, output_dir="/content/ComfyUI/output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.png"

    frame = (image.cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(frame).save(output_path)

    return output_path

def generate_video(
    image_path: str = None,
    positive_prompt: str = "a cute anime girl with massive fennec ears and a big fluffy tail wearing a maid outfit turning around",
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    width: int = 832,
    height: int = 480,
    seed: int = 82628696717253,
    steps: int = 20,
    cfg_scale: float = 1.0,
    sampler_name: str = "uni_pc",
    scheduler: str = "simple",
    frames: int = 33,
    fps: int = 16,
    output_format: str = "mp4",
    device_type: str = "CPU"
):
    # Set the device based on input
    device = set_device(device_type)

    with torch.inference_mode():
        print("Loading Text_Encoder...")
        clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0].to(device)

        positive = clip_encode_positive.encode(clip, positive_prompt)[0].to(device)
        negative = clip_encode_negative.encode(clip, negative_prompt)[0].to(device)

        del clip
        torch.cuda.empty_cache()
        gc.collect()

        if image_path is None:
            print("Please upload an image file:")
            # Handle image upload (from Colab)
            # You could integrate this function if using Google Colab
            image_path = upload_image()

        if image_path is None:
            print("No image uploaded!")
        
        loaded_image = load_image.load_image(image_path)[0].to(device)
        clip_vision = clip_vision_loader.load_clip("clip_vision_h.safetensors")[0].to(device)
        clip_vision_output = clip_vision_encode.encode(clip_vision, loaded_image, "none")[0].to(device)

        del clip_vision
        torch.cuda.empty_cache()
        gc.collect()

        print("Loading VAE...")
        vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0].to(device)

        positive_out, negative_out, latent = wan_image_to_video.encode(
            positive, negative, vae, width, height, frames, 1, loaded_image, clip_vision_output
        )

        print("Loading Unet Model...")
        if useQ6:
            model = unet_loader.load_unet("wan2.1-i2v-14b-480p-Q6_K.gguf")[0].to(device)
        else:
            model = unet_loader.load_unet("wan2.1-i2v-14b-480p-Q4_0.gguf")[0].to(device)
        model = model_sampling.patch(model, 8)[0].to(device)

        print("Generating video...")
        sampled = ksampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_out,
            negative=negative_out,
            latent_image=latent
        )[0].to(device)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        try:
            print("Decoding latents...")
            decoded = vae_decode.decode(vae, sampled)[0].to(device)

            del vae
            torch.cuda.empty_cache()
            gc.collect()

            output_path = ""
            if frames == 1:
                print("Single frame detected - saving as PNG image...")
                output_path = save_as_image(decoded[0], "ComfyUI")
                # print(f"Image saved as PNG: {output_path}")

                display(IPImage(filename=output_path, width=300))

            else:
                print(f"Saving as {output_format.upper()}...")
                if output_format == "mp4":
                    output_path = save_as_mp4(decoded, "ComfyUI", fps)
                elif output_format == "webm":
                    output_path = save_as_webm(decoded, "ComfyUI", fps)

            print(f"Video generated: {output_path}")
            return display_video(output_path)

        except Exception as e:
            print(f"Error decoding latents: {str(e)}")
            raise e


# Gradio Interface Function
def gradio_interface(
    image: gr.Image,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    scheduler: str,
    frames: int,
    fps: int,
    output_format: str,
    device_type: str  # Add the device type parameter
):
    return generate_video(
        image_path=image,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler,
        frames=frames,
        fps=fps,
        output_format=output_format,
        device_type=device_type  # Pass the device type to the video generation function
    )

# Gradio Interface Setup
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(value="a cute anime girl with massive fennec ears and a big fluffy tail wearing a maid outfit turning around", label="Positive Prompt"),
        gr.Textbox(value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", label="Negative Prompt"),
        gr.Slider(minimum=128, maximum=1024, step=1, value=832, label="Width"),
        gr.Slider(minimum=128, maximum=1024, step=1, value=480, label="Height"),
        gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Steps"),
        gr.Slider(minimum=1, maximum=20, step=0.1, value=1.0, label="CFG Scale"),
        gr.Textbox(value="uni_pc", label="Sampler Name"),
        gr.Textbox(value="simple", label="Scheduler"),
        gr.Slider(minimum=1, maximum=120, step=1, value=33, label="Frames"),
        gr.Slider(minimum=1, maximum=60, step=1, value=16, label="FPS"),
        gr.Radio(choices=["mp4", "webm"], value="mp4", label="Output Format"),
        gr.Radio(choices=["CPU", "DirectML", "ZLUDA"], value="CPU", label="Device Type")  # New input for device type
    ],
    outputs=gr.HTML(label="Generated Video"),
    live=True
)

iface.launch(share=True)
