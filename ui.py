import os
import cv2
import tempfile
import gradio as gr
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import Iterable

from transformers import (
    Sam3Model, Sam3Processor,
    Sam3TrackerModel, Sam3TrackerProcessor
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using compute device: {device}")

print("⏳ Loading SAM3 Models permanently into memory...")

try:
    # 1. Load Image Segmentation Model (Text)
    print("   ... Loading Image Text Model")
    IMG_MODEL = Sam3Model.from_pretrained("bodhicitta/sam3").to(device)
    IMG_PROCESSOR = Sam3Processor.from_pretrained("bodhicitta/sam3")

    # 2. Load Image Tracker Model (Click)
    print("   ... Loading Image Tracker Model")
    TRK_MODEL = Sam3TrackerModel.from_pretrained("bodhicitta/sam3").to(device)
    TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained("bodhicitta/sam3")

    print("✅ All Models loaded successfully!")

except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODELS: {e}")
    IMG_MODEL = None
    IMG_PROCESSOR = None
    TRK_MODEL = None
    TRK_PROCESSOR = None
    VID_MODEL = None
    VID_PROCESSOR = None


# --- UTILS ---
def apply_mask_overlay(base_image, mask_data, opacity=0.5):
    """Draws segmentation masks on top of an image."""
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")
    
    if mask_data is None or len(mask_data) == 0:
        return base_image.convert("RGB")
        
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.cpu().numpy()
    mask_data = mask_data.astype(np.uint8)
    
    # Handle dimensions
    if mask_data.ndim == 4: mask_data = mask_data[0] 
    if mask_data.ndim == 3 and mask_data.shape[0] == 1: mask_data = mask_data[0]
    
    num_masks = mask_data.shape[0] if mask_data.ndim == 3 else 1
    if mask_data.ndim == 2:
        mask_data = [mask_data]
        num_masks = 1

    try:
        color_map = matplotlib.colormaps["rainbow"].resampled(max(num_masks, 1))
    except AttributeError:
        import matplotlib.cm as cm
        color_map = cm.get_cmap("rainbow").resampled(max(num_masks, 1))
        
    rgb_colors = [tuple(int(c * 255) for c in color_map(i)[:3]) for i in range(num_masks)]
    composite_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    
    for i, single_mask in enumerate(mask_data):
        mask_bitmap = Image.fromarray((single_mask * 255).astype(np.uint8))
        if mask_bitmap.size != base_image.size:
            mask_bitmap = mask_bitmap.resize(base_image.size, resample=Image.NEAREST)
        
        fill_color = rgb_colors[i]
        color_fill = Image.new("RGBA", base_image.size, fill_color + (0,))
        mask_alpha = mask_bitmap.point(lambda v: int(v * opacity) if v > 0 else 0)
        color_fill.putalpha(mask_alpha)
        composite_layer = Image.alpha_composite(composite_layer, color_fill)
        
    return Image.alpha_composite(base_image, composite_layer).convert("RGB")

def draw_points_on_image(image, points):
    """Draws red dots on the image to indicate click locations."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    for pt in points:
        x, y = pt
        r = 8 # Radius of point
        draw.ellipse((x-r, y-r, x+r, y+r), fill="red", outline="white", width=4)
    
    return draw_img

def run_image_segmentation(source_img, text_query, conf_thresh=0.5):
    if IMG_MODEL is None or IMG_PROCESSOR is None:
        raise gr.Error("Models failed to load on startup.")
        
    if source_img is None or not text_query:
        raise gr.Error("Please provide an image and a text prompt.")
    
    try:
        pil_image = source_img.convert("RGB")
        model_inputs = IMG_PROCESSOR(images=pil_image, text=text_query, return_tensors="pt").to(device)

        with torch.no_grad():
            inference_output = IMG_MODEL(**model_inputs)

        processed_results = IMG_PROCESSOR.post_process_instance_segmentation(
            inference_output,
            threshold=conf_thresh,
            mask_threshold=0.5,
            target_sizes=model_inputs.get("original_sizes").tolist()
        )[0]

        annotation_list = []
        raw_masks = processed_results['masks'].cpu().numpy()
        raw_scores = processed_results['scores'].cpu().numpy()
        
        for idx, mask_array in enumerate(raw_masks):
            label_str = f"{text_query} ({raw_scores[idx]:.2f})"
            annotation_list.append((mask_array, label_str))
            
        return (pil_image, annotation_list)

    except Exception as e:
        raise gr.Error(f"Error during image processing: {e}")

def run_image_click_gpu(input_image, x, y, points_state, labels_state):
    if TRK_MODEL is None or TRK_PROCESSOR is None:
        raise gr.Error("Tracker Model failed to load.")
    
    if input_image is None: return input_image, [], []
    if points_state is None: points_state = []; labels_state = []
    
    # Append new point
    points_state.append([x, y])
    labels_state.append(1) # 1 indicates a positive click (foreground)

    try:
        # Prepare inputs format: [Batch, Point_Group, Point_Idx, Coord]
        input_points = [[points_state]] 
        input_labels = [[labels_state]]
        
        inputs = TRK_PROCESSOR(images=input_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # multimask_output=True usually helps with ambiguity, but let's default to best mask for simplicity here
            outputs = TRK_MODEL(**inputs, multimask_output=False)
            
        # Post process
        masks = TRK_PROCESSOR.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"], binarize=True)[0]
        
        # Overlay mask
        # masks shape is [1, 1, H, W] for single object tracking
        final_img = apply_mask_overlay(input_image, masks[0])
        
        # Draw the visual points on top
        final_img = draw_points_on_image(final_img, points_state)
        
        return final_img, points_state, labels_state

    except Exception as e:
        print(f"Tracker Error: {e}")
        return input_image, points_state, labels_state

def image_click_handler(image, evt: gr.SelectData, points_state, labels_state):
    # Wrapper to handle the Gradio select event
    x, y = evt.index
    return run_image_click_gpu(image, x, y, points_state, labels_state)

def calc_timeout_duration(vid_file, *args):
    return args[-1] if args else 60

custom_css="""
#col-container { margin: 0 auto; max-width: 1200px; }
#main-title h1 { font-size: 2.2em !important; }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        with gr.Tabs():
            with gr.Tab("Image Segmentation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Upload Image", type="pil", height=350)
                        txt_prompt_img = gr.Textbox(label="Text Prompt", placeholder="e.g., cat, face, car wheel")
                        with gr.Accordion("Advanced Settings", open=False):
                            conf_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Confidence Threshold")
                        
                        btn_process_img = gr.Button("Segment Image", variant="primary")

                    with gr.Column(scale=1.5):
                        image_result = gr.AnnotatedImage(label="Segmented Result", height=410)
        
                        btn_process_img.click(
                            fn=run_image_segmentation, 
                            inputs=[image_input, txt_prompt_img, conf_slider], 
                            outputs=[image_result]
                        )
   
            with gr.Tab("Image Click Segmentation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_click_input = gr.Image(type="pil", label="Upload Image", interactive=True, height=450)
                        
                        with gr.Row():
                            img_click_clear = gr.Button("Clear Points & Reset", variant="primary")
                        
                        st_click_points = gr.State([])
                        st_click_labels = gr.State([])

                    with gr.Column(scale=1):
                        img_click_output = gr.Image(type="pil", label="Result Preview", height=450, interactive=False)
                
                img_click_input.select(
                    image_click_handler,
                    inputs=[img_click_input, st_click_points, st_click_labels],
                    outputs=[img_click_output, st_click_points, st_click_labels]
                )
                
                img_click_clear.click(
                    lambda: (None, [], []),
                    outputs=[img_click_output, st_click_points, st_click_labels]
                )

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=gr.themes.Citrus(), ssr_mode=False, mcp_server=True, show_error=True)