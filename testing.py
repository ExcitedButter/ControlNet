import torch
from torchvision import transforms
from PIL import Image
import json
from pytorch_fid import fid_score
from pathlib import Path
from share import *
import config
import cv2
import einops
import gradio as gr
import numpy as np
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import clip

apply_canny = CannyDetector()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

# Load the prompts
with open('/home/zhicao/output/prompt.json', 'r') as f:
    prompt_data = json.load(f)

real_images_path = Path('/home/zhicao/output')
source_images_path = Path('/home/zhicao/output')
fake_images_path = Path('/home/zhicao/output/generate')

# Make sure the directory for fake images exists
fake_images_path.mkdir(parents=True, exist_ok=True)

# Process each entry in the JSON file
for entry in prompt_data:
    source_image_path = source_images_path / entry['source']
    source_image = Image.open(source_image_path).convert("RGB")
    source_image = np.array(source_image, dtype=np.uint8)
    # Modify these parameters as per your model's requirements
    num_samples = 1
    image_resolution = 256
    ddim_steps = 50
    guess_mode = False
    strength = 0.8
    scale = 5.0
    seed = -1  # Use random seed
    eta = 0.0
    low_threshold = 50
    high_threshold = 150

    # Assuming 'a_prompt' and 'n_prompt' are defined somewhere
    a_prompt = "additional prompt info"
    n_prompt = "negative prompt info"

    # Call the process function
    generated_images = process(
        source_image, entry['prompt'], a_prompt, n_prompt, num_samples,
        image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta,
        low_threshold, high_threshold
    )

    # Assuming the first image in the results is the generated image
    generated_image = generated_images[1]  # Skip the edge-detected image

    # Convert numpy array back to PIL Image and save
    generated_image_pil = Image.fromarray(generated_image)
    save_path = fake_images_path / entry['source']
    generated_image_pil.save(save_path)


# calculate FID
real_images_path1 = Path('/home/zhicao/output/target')
fake_images_path1 = Path('/home/zhicao/output/generate/source')
fid_value = fid_score.calculate_fid_given_paths([str(real_images_path1), str(fake_images_path1)],
                                                batch_size=5, device=device, dims=2048)
print(f"FID Score: {fid_value}")



# Calculate CLIP-score for each generated image and its prompt
clip_model, preprocess = clip.load("ViT-B/32", device=device)
def clip_score(image, text, clip_model):
    text_inputs = clip.tokenize([text]).to(device)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

    cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features).cpu().numpy()

    return cosine_similarity

clip_values = []
for entry in prompt_data:
    image_path = fake_images_path / entry['source']
    generated_image = Image.open(image_path).convert("RGB")
    
    text_prompt = entry['prompt']
    clip_value = clip_score(generated_image, text_prompt, clip_model)
    clip_values.append(clip_value)

# Calculate average CLIP-score
average_clip_score = sum(clip_values) / len(clip_values)
print(f"Average CLIP Score: {average_clip_score}")