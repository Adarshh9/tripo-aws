import logging
import os
import time

import numpy as np
# from PIL import  Image, ImageOps
# import numpy as np
import torch
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import save_video
from tsr.bake_texture import bake_texture


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


def initialize_model(pretrained_model_name_or_path="stabilityai/TripoSR",
                     chunk_size=8192,
                     device="cuda:0" if torch.cuda.is_available() else "cpu"):
    timer.start("Initializing model")
    model = TSR.from_pretrained(
        pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)
    timer.end("Initializing model")
    return model


def process_image(image_path, output_dir, no_remove_bg=True, foreground_ratio=0.85):
    timer.start("Processing image")

    if no_remove_bg:
        rembg_session = None
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = remove_background(image_path)
        
        # Save the processed image
        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, "processed_input.png"))

    timer.end("Processing image")
    return image


def run_model(model, image, output_dir, device="cuda:0" if torch.cuda.is_available() else "cpu", render=False, mc_resolution=256, model_save_format='obj', bake_texture_flag=False, texture_resolution=2048):
    logging.info("Running model...")

    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, f"render_{ri:03d}.png"))
        save_video(
            render_images[0], os.path.join(output_dir, "render.mp4"), fps=30
        )
        timer.end("Rendering")

    timer.start("Extracting mesh")
    meshes = model.extract_mesh(scene_codes, not bake_texture_flag, resolution=mc_resolution)
    timer.end("Extracting mesh")

    out_mesh_path = os.path.join(output_dir, f"mesh.{model_save_format}")
    if bake_texture_flag:
        out_texture_path = os.path.join(output_dir, "texture.png")

        timer.start("Baking texture")
        bake_output = bake_texture(meshes[0], model, scene_codes[0], texture_resolution)
        timer.end("Baking texture")

        timer.start("Exporting mesh and texture")
        xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
        timer.end("Exporting mesh and texture")
    else:
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")

    return out_mesh_path


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
timer = Timer()
