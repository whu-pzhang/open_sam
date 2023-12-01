import os
import warnings

import gradio as gr
import lightning as L
import numpy as np
from lightning.app.components.serve import ServeGradio
from PIL import Image

from lang_sam import LangSAM
from lang_sam import SAM_MODELS

from open_sam.builder import build_sam
from open_sam.utils import load_image, draw_image

warnings.filterwarnings("ignore")


class LitGradio(ServeGradio):

    inputs = [
        gr.Dropdown(choices=['tiny', 'base', 'large', 'huge'],
                    label="SAM model",
                    value="tiny"),
        gr.Slider(0, 1, value=0.3, label="Box threshold"),
        gr.Slider(0, 1, value=0.25, label="Text threshold"),
        gr.Image(type="filepath", label='Image'),
        gr.Textbox(lines=1, label="Text Prompt"),
    ]
    outputs = [gr.outputs.Image(type="pil", label="Output Image")]

    examples = [
        [
            'tiny',
            0.36,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
            "kiwi",
        ],
        [
            'tiny',
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
            "wheel",
        ],
        [
            'tiny',
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
            "food",
        ],
    ]

    def __init__(self, sam_type="tiny"):
        super().__init__()
        self.ready = False
        self.sam_type = sam_type

    def predict(self, sam_type, box_threshold, image_path):
        print("Predicting... ", sam_type, box_threshold, image_path)
        if sam_type != self.model.sam_type:
            self.model.build_sam(sam_type)
        image_pil = load_image(image_path)
        masks, boxes, phrases, logits = self.model.predict(
            image_pil, box_threshold)
        labels = [
            f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)
        ]
        image_array = np.asarray(image_pil)
        image = draw_image(image_array, masks, boxes, labels)
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        return image

    def build_model(self, sam_type="tiny"):
        model = LangSAM(sam_type)
        self.ready = True
        return model


app = L.LightningApp(LitGradio())
