import time
import streamlit as st
from PIL import ExifTags, Image, ImageFile
from streamlit_scroll_to_top import scroll_to_here
import os
import json
import uuid
from datetime import datetime
import zipfile
import io
import re
from io import BytesIO
from glob import iglob
import torch
from einops import rearrange
from fire import Fire
from torchvision import transforms
from transformers import pipeline

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
import logging

logging.basicConfig(level=logging.DEBUG)

NSFW_THRESHOLD = 0.85
HISTORY_DIR = "history"
    
ImageFile.LOAD_TRUNCATED_IMAGES = True

@st.cache_resource()
def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, t5, clip, nsfw_classifier

@torch.inference_mode()
def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
):
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    HISTORY_FILE = os.path.join(HISTORY_DIR, "generation_history.json")

    st.set_page_config(
        page_title="recreategoods",
        page_icon="🪡",
        layout="wide"
    )

    torch_device = torch.device(device)
    
    if 'scroll_to_top' not in st.session_state:
        st.session_state.scroll_to_top = False

    if st.session_state.scroll_to_top:
        scroll_to_here(0, key='top')
        st.session_state.scroll_to_top = False

    def load_history():
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []

    def save_history(history):
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    def save_generation(input_image, prompt, output_image):
        # Generate unique ID for this generation
        gen_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Get original filename without extension
        original_filename = st.session_state.last_uploaded_file
        base_filename = os.path.splitext(original_filename)[0]
        
        # Get input image extension
        input_ext = os.path.splitext(original_filename)[1].lstrip('.')
        if not input_ext:
            input_ext = "jpg"  # Default to jpg
        
        # Save input image
        input_filename = f"{base_filename}_{gen_id}_input.{input_ext}"
        input_path = os.path.join(HISTORY_DIR, input_filename)
        input_image.save(input_path)
        
        # Save output image
        output_filename = f"{base_filename}_{gen_id}_output.{input_ext}"
        output_path = os.path.join(HISTORY_DIR, output_filename)
        output_image.save(output_path)
        
        entry = {
            "id": gen_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "model": selected_model,
            "input_image": input_filename,
            "output_image": output_filename,
            "original_filename": original_filename
        }
        
        history = load_history()
        history.append(entry)
        save_history(history)
        
        return entry

    st.markdown("""
        <style>
        .stMainBlockContainer {
            padding-top: 50px !important;
        }
        .title {
            font-size: 56px !important;
            font-weight: 700;
            letter-spacing: 2px;
            padding: 20px 0 50px !important;
        }
        .step-header {
            font-size: 20px;
            font-weight: bold;
            color: #1E88E5;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        /* Hide file uploader name */
        .stFileUploaderFile, .st-emotion-cache-12xsiil {
            display: none !important;
        }
        .history-entry {
            margin-bottom: 20px;
        }
        .stButton p {
            font-size: 14px;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .stMarkdown {
            height: 100%;
        }
        .stHorizontalBlock:has(.history) ~ .stHorizontalBlock > .stColumn:has(.stButton) {
            background: rgb(240, 242, 246);
            border-radius: 0.5rem;
            padding: 0 15px 15px;
            margin-bottom: 10px;
            max-width: 100%;
        }
        .history-prompt {
            font-size: 12px;
            color: #666;
            border-radius: 4px;
            height: 100%;
            align-items: center;
            line-height: 16px;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 15px;
        }
        .history-header {
            font-size: 14px;
            margin-bottom: 8px;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .refine-button {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-top: 10px;
        }
        .refine-button p {
            margin: 0;
        }
        div.stButton button {
            min-width: 100px;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        div[data-testid="stHorizontalBlock"] div.stButton button {
            text-align: left;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            display: block;
            padding-right: 20px;
        }
        /* Keep action buttons centered */
        div[data-testid="column"]:has(div.stButton) button {
            text-align: center;
            min-width: 100px;
        }
        </style>
        <h1 class="title">r e c r e a t e g o o d s</h1>
    """, unsafe_allow_html=True)

    # Initialize session state for uploaded image if it doesn't exist
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    # Initialize session state for delete confirmation
    if 'show_delete_confirmation' not in st.session_state:
        st.session_state.show_delete_confirmation = False

    # Initialize session state for download confirmation
    if 'show_download_confirmation' not in st.session_state:
        st.session_state.show_download_confirmation = False

    def delete_all_history():
        # Clear the history file
        save_history([])
        # Delete all files in history directory except the history.json
        for filename in os.listdir(HISTORY_DIR):
            if filename != "generation_history.json":
                file_path = os.path.join(HISTORY_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"Error deleting {filename}: {e}")
        st.session_state.show_delete_confirmation = False
        st.rerun()

    def create_download_zip():
        # Create a BytesIO object to store the zip file
        zip_buffer = io.BytesIO()
        
        # Create a new zip file
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add history.json
            zip_file.write(HISTORY_FILE, "history.json")
            
            # Add all image files
            for filename in os.listdir(HISTORY_DIR):
                if filename != "generation_history.json":
                    file_path = os.path.join(HISTORY_DIR, filename)
                    if os.path.isfile(file_path):
                        zip_file.write(file_path, filename)
        
        # Reset buffer position
        zip_buffer.seek(0)
        return zip_buffer

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col1:
        st.markdown('<p class="step-header">Step 1: Upload Your Image of the Garment</p>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, caption="Input Image", use_container_width=True)
        else:
            placeholder_height = 300
            st.markdown(
                f"""
                <div style="
                    height: {placeholder_height}px;
                    border: 2px dashed #cccccc;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    color: #666666;
                    padding: 20px;
                ">
                    Input image will appear here
                </div>
                """,
                unsafe_allow_html=True
            )
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None and (st.session_state.uploaded_image is None or uploaded_file.name != getattr(st.session_state, 'last_uploaded_file', None)):
            try:
                # Save the file name to track changes
                st.session_state.last_uploaded_file = uploaded_file.name
                # Convert to PIL Image
                image = Image.open(uploaded_file)
                # Convert to RGB if necessary
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                # Create a copy of the image to ensure it's loaded
                image_copy = image.copy()
                st.session_state.uploaded_image = image_copy
                st.rerun()
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    # Column 2: Edit Instructions and Model Selection
    with col2:
        st.markdown('<p class="step-header">Step 2: Enter Your Edit Instructions</p>', unsafe_allow_html=True)
        if "prompt" not in st.session_state:
            st.session_state.prompt = ""
        prompt = st.text_area("Describe how you want to modify the garment", 
                            value=st.session_state.prompt,  # Use the session state value
                            height=100)
        st.session_state.prompt = prompt
        
        st.markdown('<p class="step-header">Step 3: Choose Your Model</p>', unsafe_allow_html=True)
        model_options = list(configs.keys())
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = model_options[0]
        selected_model = st.selectbox("Select the AI model to use", 
                                    model_options,
                                    index=model_options.index(st.session_state.selected_model))
        st.session_state.selected_model = selected_model
        
        st.markdown('<p class="step-header">Step 4: Generate</p>', unsafe_allow_html=True)
        if st.button("Generate Image", use_container_width=True):
            if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
                st.error("Please upload an image first!")
            elif not prompt:
                st.error("Please enter edit instructions!")
            else:
                with st.spinner("Generating image..."):
                    name = st.session_state.selected_model
                    is_schnell = name == "flux-schnell"

                    model, ae, t5, clip, nsfw_classifier = get_models(
                        name,
                        device=torch_device,
                        offload=offload,
                        is_schnell=is_schnell,
                    )

                    # image = Image.open(st.session_state.uploaded_image).convert("RGB")
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: 2.0 * x - 1.0),
                        ]
                    )
                    img: torch.Tensor = transform(st.session_state.uploaded_image)
                    init_image = img[None, ...]
                    image2image_strength = 0.8 # TODO: st.number_input("Noising strength", min_value=0.0, max_value=1.0, value=0.8)
                    if init_image is not None:
                        h, w = init_image.shape[-2:]
                        st.write(f"Got image of size {w}x{h} ({h*w/1e6:.2f}MP)")
                    resize_img = True # TODO: st.checkbox("Resize image", False)

                    # allow for packing and conversion to latent space
                    width = 1360 # TODO: int(
                    #     16 * (st.number_input("Width", min_value=128, value=1360, step=16, disabled=not resize_img) // 16)
                    # )
                    height = 768 # TODO: int(
                    #     16 * (st.number_input("Height", min_value=128, value=768, step=16, disabled=not resize_img) // 16)
                    # )
                    num_steps = 4 if is_schnell else 50  # TODO: int(st.number_input("Number of steps", min_value=1, value=(4 if is_schnell else 50)))
                    guidance = 3.5 # TODO: float(st.number_input("Guidance", min_value=1.0, value=3.5, disabled=is_schnell))
                    seed_str = "123456" # TODO: st.text_input("Seed", disabled=is_schnell)
                    if seed_str.isdecimal():
                        seed = int(seed_str)
                    else:
                        st.info("No seed set, set to positive integer to enable")
                        seed = None
                    save_samples = True # TODO: st.checkbox("Save samples?", not is_schnell)
                    add_sampling_metadata = True # TODO: st.checkbox("Add sampling parameters to metadata?", True)

                    output_name = os.path.join(output_dir, "img_{idx}.jpg")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        idx = 0
                    else:
                        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
                        if len(fns) > 0:
                            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                        else:
                            idx = 0

                    rng = torch.Generator(device="cpu")

                    if "seed" not in st.session_state:
                        st.session_state.seed = rng.seed()

                    def increment_counter():
                        st.session_state.seed += 1

                    def decrement_counter():
                        if st.session_state.seed > 0:
                            st.session_state.seed -= 1

                    opts = SamplingOptions(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_steps=num_steps,
                        guidance=guidance,
                        seed=seed,
                    )  

                    if name == "flux-schnell":
                        pass
                        # TODO: cols = st.columns([5, 1, 1, 5])
                        # with cols[1]:
                        #     st.button("↩", on_click=increment_counter)
                        # with cols[2]:
                        #     st.button("↪", on_click=decrement_counter)
                    if is_schnell or True: # TODO: st.button("Sample"):
                        if is_schnell:
                            opts.seed = st.session_state.seed
                        elif opts.seed is None:
                            opts.seed = rng.seed()
                        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
                        t0 = time.perf_counter()

                        if resize_img:
                            init_image = torch.nn.functional.interpolate(init_image, (opts.height, opts.width))
                        else:
                            h, w = init_image.shape[-2:]
                            init_image = init_image[..., : 16 * (h // 16), : 16 * (w // 16)]
                            opts.height = init_image.shape[-2]
                            opts.width = init_image.shape[-1]
                        if offload:
                            ae.encoder.to(torch_device)
                        init_image = ae.encode(init_image.to(torch_device))
                        if offload:
                            ae = ae.cpu()
                            torch.cuda.empty_cache()

                    # prepare input
                    x = get_noise(
                        1,
                        opts.height,
                        opts.width,
                        device=torch_device,
                        dtype=torch.bfloat16,
                        seed=opts.seed,
                    )
                    # divide pixel space by 16**2 to account for latent space conversion
                    timesteps = get_schedule(
                        opts.num_steps,
                        (x.shape[-1] * x.shape[-2]) // 4,
                        shift=(not is_schnell),
                    )
                    t_idx = int((1 - image2image_strength) * num_steps)
                    t = timesteps[t_idx]
                    timesteps = timesteps[t_idx:]
                    x = t * x + (1.0 - t) * init_image.to(x.dtype)

                    if offload:
                        t5, clip = t5.to(torch_device), clip.to(torch_device)
                    inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)

                    # offload TEs to CPU, load model to gpu
                    if offload:
                        t5, clip = t5.cpu(), clip.cpu()
                        torch.cuda.empty_cache()
                        model = model.to(torch_device)

                    # denoise initial noise
                    x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

                    # offload model, load autoencoder to gpu
                    if offload:
                        model.cpu()
                        torch.cuda.empty_cache()
                        ae.decoder.to(x.device)

                    # decode latents to pixel space
                    x = unpack(x.float(), opts.height, opts.width)
                    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                        x = ae.decode(x)

                    if offload:
                        ae.decoder.cpu()
                        torch.cuda.empty_cache()

                    t1 = time.perf_counter()

                    fn = output_name.format(idx=idx)
                    print(f"Done in {t1 - t0:.1f}s.")
                    # bring into PIL format and save
                    x = x.clamp(-1, 1)
                    x = embed_watermark(x.float())
                    x = rearrange(x[0], "c h w -> h w c")

                    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

                    if nsfw_score < NSFW_THRESHOLD:
                        buffer = BytesIO()
                        exif_data = Image.Exif()
                        if init_image is None:
                            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                        else:
                            exif_data[ExifTags.Base.Software] = "AI generated;img2img;flux"
                        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                        exif_data[ExifTags.Base.Model] = name
                        if add_sampling_metadata:
                            exif_data[ExifTags.Base.ImageDescription] = prompt
                        img.save(buffer, format="jpeg", exif=exif_data, quality=95, subsampling=0)

                        img_bytes = buffer.getvalue()
                        if save_samples:
                            print(f"Saving {fn}")
                            with open(fn, "wb") as file:
                                file.write(img_bytes)
                            idx += 1

                        st.session_state["samples"] = {
                            "prompt": opts.prompt,
                            "img": img,
                            "seed": opts.seed,
                            "bytes": img_bytes,
                        }
                        opts.seed = None
                    else:
                        st.warning("Your generated image may contain NSFW content.")
                        st.session_state["samples"] = None

                    samples = st.session_state.get("samples", None)
                    if samples is not None:
                        st.session_state.generated_image = samples["img"]
                    
                    save_generation(
                        input_image=st.session_state.uploaded_image,
                        prompt=prompt,
                        output_image=samples["img"]
                    )

    with col3:
        st.markdown('<p class="step-header">Output</p>', unsafe_allow_html=True)
        samples = st.session_state.get("samples", None)
        if samples is not None:
            st.image(samples["img"], caption="Generated Image", use_container_width=True)
            st.download_button(
                "Download full-resolution",
                samples["bytes"],
                file_name="generated.jpg",
                mime="image/jpg",
            )
            st.write(f"Seed: {samples['seed']}")

            refine_col = st.columns([2, 1, 2])[1]  # Center the button
            with refine_col:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:has(div.refine-button) {
                        display: flex;
                        justify-content: center;
                    }
                    </style>
                    """, 
                    unsafe_allow_html=True
                )
                if st.button("↩ Refine", key="refine_button"):
                    st.session_state.uploaded_image = samples["img"]
                    del st.session_state.samples
                    st.session_state.scroll_to_top = True
                    st.rerun()
        else:
            placeholder_height = 300
            st.markdown(
                f"""
                <div style="
                    height: {placeholder_height}px;
                    border: 2px dashed #cccccc;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    color: #666666;
                    padding: 20px;
                ">
                    Generated image will appear here
                </div>
                """,
                unsafe_allow_html=True
            )

    # History Section
    st.markdown("---")

    history = load_history()
    history.reverse()  # Most recent first

    header_col, buttons_col = st.columns([0.6, 0.4])

    with header_col:
        st.markdown('<p class="step-header history">History</p>', unsafe_allow_html=True)

    with buttons_col:
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download", key="download", disabled=len(history) == 0):
                st.session_state.show_download_confirmation = True
        with col2:
            if st.button("Delete", key="delete", disabled=len(history) == 0):
                st.session_state.show_delete_confirmation = True

    if st.session_state.show_download_confirmation:
        with st.container():
            st.markdown("""
                <style>
                .download-confirmation {
                    background-color: #e8f5e9;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                </style>
                <div class="download-confirmation">
                    <h3>📥 Download History</h3>
                    <p>Download a zip file containing all history entries and their associated images.</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                zip_buffer = create_download_zip()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if st.download_button(
                    label="Yes, Download",
                    data=zip_buffer,
                    file_name=f"history_{timestamp}.zip",
                    mime="application/zip",
                    key="confirm_download"
                ):
                    st.session_state.show_download_confirmation = False
                    st.rerun()
            with col2:
                if st.button("Cancel", key="cancel_download"):
                    st.session_state.show_download_confirmation = False
                    st.rerun()

    if st.session_state.show_delete_confirmation:
        with st.container():
            st.markdown("""
                <style>
                .delete-confirmation {
                    background-color: #ffebee;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                </style>
                <div class="delete-confirmation">
                    <h3>⚠️ Delete All History?</h3>
                    <p>This action cannot be undone. All history entries and their associated images will be permanently deleted.</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete All", key="confirm_delete"):
                    delete_all_history()
            with col2:
                if st.button("Cancel", key="cancel_delete"):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()

    # Display history in rows of 3
    for i in range(0, len(history), 3):
        row_entries = history[i:i+3]
        cols = st.columns(3)
        
        for col_idx, entry in enumerate(row_entries):
            with cols[col_idx]:
                dt = datetime.fromisoformat(entry["timestamp"])
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                
                full_header = f"#{len(history)-i-col_idx} - {formatted_time} ({entry.get('model', 'Unknown Model')})"
                
                st.markdown(f"""
                    <style>
                    div[data-testid="stHorizontalBlock"] div.stButton button {{
                        cursor: pointer;
                    }}
                    </style>
                """, unsafe_allow_html=True)
                
                if st.button(full_header, key=f"history_{entry['id']}", use_container_width=True, help=full_header):
                    # Restore the history entry
                    st.session_state.uploaded_image = Image.open(os.path.join(HISTORY_DIR, entry["input_image"]))
                    st.session_state.last_uploaded_file = entry["original_filename"]
                    st.session_state["prompt"] = entry["prompt"]
                    st.session_state["selected_model"] = entry.get("model", "Stable Diffusion")
                    st.session_state.generated_image = Image.open(os.path.join(HISTORY_DIR, entry["output_image"]))
                    st.session_state.scroll_to_top = True
                    st.rerun()
                
                img1_col, prompt_col, img2_col = st.columns(3)
                
                with img1_col:
                    input_path = os.path.join(HISTORY_DIR, entry["input_image"])
                    if os.path.exists(input_path):
                        input_img = Image.open(input_path)
                        st.image(input_img, use_container_width=True)
                
                with prompt_col:
                    st.markdown(
                        f"""<div class="history-prompt" title="{entry['prompt']}">{entry['prompt']}</div>""",
                        unsafe_allow_html=True
                    )
                
                with img2_col:
                    output_path = os.path.join(HISTORY_DIR, entry["output_image"])
                    if os.path.exists(output_path):
                        output_img = Image.open(output_path)
                        st.image(output_img, use_container_width=True)

def app():
    Fire(main)

if __name__ == "__main__":
    app()