import sys
import importlib.util
from unittest.mock import MagicMock
import torch

# Fix for remote code flash-attn bug
class FakeFlashAttnSpec:
    name = 'flash_attn'
    loader = None
    origin = None
    submodule_search_locations = []
    
fake_spec = FakeFlashAttnSpec()
flash_attn_mock = MagicMock()
flash_attn_mock.__spec__ = fake_spec
flash_attn_mock.__version__ = "0.0.0"

sys.modules['flash_attn'] = flash_attn_mock
sys.modules['flash_attn.flash_attn_interface'] = MagicMock()
sys.modules['flash_attn.bert_padding'] = MagicMock()

_original_find_spec = importlib.util.find_spec
def _patched_find_spec(name, package=None):
    if name == 'flash_attn' or name.startswith('flash_attn.'):
        return fake_spec
    return _original_find_spec(name, package)
importlib.util.find_spec = _patched_find_spec

# --- Imports ---
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
import copy
from PIL import Image, ImageDraw
import random
import numpy as np

# --- Device & Dtype Setup ---
# Checks for Intel Arc GPU via PyTorch XPU, otherwise uses CPU with bfloat16
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
    dtype = torch.float16
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    # bfloat16 is heavily accelerated on Intel Core Ultra CPUs
    dtype = torch.bfloat16

print(f"Running inference on: {device} with dtype: {dtype}")

MODEL_ID = 'microsoft/Florence-2-large'
REVISION = '00d2f1570b00c6dea5df998f5635db96840436bc'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    trust_remote_code=True,
    attn_implementation="sdpa",
    revision=REVISION
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True
)

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def run_example(task_prompt, image, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Send inputs to device and appropriate precision
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        for k, v in inputs.items()
    }
    
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=1,  # 1 instead of 3 reduces computation by ~3x
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

# Direct PIL drawing: Replaces slow Matplotlib subplots and removes memory leaks
def plot_bbox(image, data):
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    for bbox, label in zip(data.get('bboxes', []), data.get('labels', [])):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 4, y1 + 2), str(label), fill="red")
    return output_image

def draw_polygons(image, prediction, fill_mask=False):
    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = color if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image

def convert_to_od_format(data):
    return {
        'bboxes': data.get('bboxes', []),
        'labels': data.get('bboxes_labels', [])
    }

def draw_ocr_bboxes(image, prediction):
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box)).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2), str(label), fill=color)
    return image

def process_image(image, task_prompt, text_input=None):
    image = Image.fromarray(image)
    
    if task_prompt in ['Caption', 'Detailed Caption', 'More Detailed Caption', 'Region to Category', 'Region to Description', 'OCR']:
        tag_map = {
            'Caption': '<CAPTION>',
            'Detailed Caption': '<DETAILED_CAPTION>',
            'More Detailed Caption': '<MORE_DETAILED_CAPTION>',
            'Region to Category': '<REGION_TO_CATEGORY>',
            'Region to Description': '<REGION_TO_DESCRIPTION>',
            'OCR': '<OCR>'
        }
        tag = tag_map[task_prompt]
        return run_example(tag, image, text_input), None

    elif task_prompt in ['Caption + Grounding', 'Detailed Caption + Grounding', 'More Detailed Caption + Grounding']:
        tag_map = {
            'Caption + Grounding': '<CAPTION>',
            'Detailed Caption + Grounding': '<DETAILED_CAPTION>',
            'More Detailed Caption + Grounding': '<MORE_DETAILED_CAPTION>'
        }
        base_tag = tag_map[task_prompt]
        results = run_example(base_tag, image)
        text_grounding = results[base_tag]
        ground_results = run_example('<CAPTION_TO_PHRASE_GROUNDING>', image, text_grounding)
        ground_results[base_tag] = text_grounding
        img_out = plot_bbox(image, ground_results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return ground_results, img_out

    elif task_prompt in ['Object Detection', 'Dense Region Caption', 'Region Proposal', 'Caption to Phrase Grounding']:
        tag_map = {
            'Object Detection': '<OD>',
            'Dense Region Caption': '<DENSE_REGION_CAPTION>',
            'Region Proposal': '<REGION_PROPOSAL>',
            'Caption to Phrase Grounding': '<CAPTION_TO_PHRASE_GROUNDING>'
        }
        tag = tag_map[task_prompt]
        results = run_example(tag, image, text_input)
        img_out = plot_bbox(image, results[tag])
        return results, img_out

    elif task_prompt in ['Referring Expression Segmentation', 'Region to Segmentation']:
        tag = '<REFERRING_EXPRESSION_SEGMENTATION>' if task_prompt == 'Referring Expression Segmentation' else '<REGION_TO_SEGMENTATION>'
        results = run_example(tag, image, text_input)
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results[tag], fill_mask=True)
        return results, output_image

    elif task_prompt == 'Open Vocabulary Detection':
        results = run_example('<OPEN_VOCABULARY_DETECTION>', image, text_input)
        bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
        img_out = plot_bbox(image, bbox_results)
        return results, img_out

    elif task_prompt == 'OCR with Region':
        results = run_example('<OCR_WITH_REGION>', image)
        output_image = copy.deepcopy(image)
        output_image = draw_ocr_bboxes(output_image, results['<OCR_WITH_REGION>'])
        return results, output_image

    return "", None

# --- UI Setup ---
single_task_list = [
    'Caption', 'Detailed Caption', 'More Detailed Caption', 'Object Detection',
    'Dense Region Caption', 'Region Proposal', 'Caption to Phrase Grounding',
    'Referring Expression Segmentation', 'Region to Segmentation',
    'Open Vocabulary Detection', 'Region to Category', 'Region to Description',
    'OCR', 'OCR with Region'
]

cascased_task_list = [
    'Caption + Grounding', 'Detailed Caption + Grounding', 'More Detailed Caption + Grounding'
]

def update_task_dropdown(choice):
    return gr.Dropdown(choices=cascased_task_list, value='Caption + Grounding') if choice == 'Cascased task' else gr.Dropdown(choices=single_task_list, value='Caption')

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Picture")
            task_type = gr.Radio(choices=['Single task', 'Cascased task'], label='Task type selector', value='Single task')
            task_prompt = gr.Dropdown(choices=single_task_list, label="Task Prompt", value="Caption")
            task_type.change(fn=update_task_dropdown, inputs=task_type, outputs=task_prompt)
            text_input = gr.Textbox(label="Text Input (optional)")
            submit_btn = gr.Button(value="Submit")
        with gr.Column():
            output_text = gr.Textbox(label="Output Text")
            output_img = gr.Image(label="Output Image")

    submit_btn.click(process_image, [input_img, task_prompt, text_input], [output_text, output_img])

demo.launch(debug=True)