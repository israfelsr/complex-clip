from abc import ABC, abstractmethod
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from peft import PeftModel
import open_clip
import os


BASE = "/leonardo_work/EUHPC_D12_071/projects/complex-clip/models/clip-vit-base-patch32"
MODEL_ROOT = "/leonardo_work/EUHPC_D12_071/projects/complex-clip/"


class ContrastiveModel(ABC):

    @abstractmethod
    def encode_text(self, text):
        """Encode text into embeddings."""
        pass

    @abstractmethod
    def encode_image(self, image):
        """Encode images into embeddings."""
        pass

    @abstractmethod
    def load_model(self, model_args, device):
        """Load the model, tokenizer and processor"""
        pass

    def encode_text(self, texts, device):
        texts = self._prepare_text(texts)
        embeddings = self._get_text_features(texts.to(device))
        return self._normalize(embeddings)

    def encode_image(self, images, device):
        images = self._prepare_image(images)
        embeddings = self._get_image_features(images.to(device))
        return self._normalize(embeddings)

    def _normalize(self, embeddings):
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    @abstractmethod
    def _prepare_text(self, texts):
        pass

    @abstractmethod
    def _prepare_image(self, images):
        pass

    @abstractmethod
    def _get_text_features(self, inputs):
        pass

    @abstractmethod
    def _get_image_features(self, inputs):
        pass


class HuggingFaceCLIP(ContrastiveModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.tag = "HF"

    def load_model(self, model_args, device):
        if model_args.lora:
            base = CLIPModel.from_pretrained(
                BASE,
                local_files_only=True,
            )
            self.model = PeftModel.from_pretrained(base, model_args.model_path)
        else:
            self.model = CLIPModel.from_pretrained(
                model_args.model_path,
                local_files_only=model_args.local_files_only,
            )
        print(f"Model loaded from: {model_args.model_path}")
        if model_args.processor_path is None:
            model_args.processor_path = model_args.model_path

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_args.processor_path, local_files_only=model_args.local_files_only
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            model_args.processor_path,
            local_files_only=model_args.local_files_only,
        )
        self.model.to(device)

    def _prepare_text(self, texts):
        return self.tokenizer(
            texts, padding="max_length", return_tensors="pt", truncation=True
        )

    def _prepare_image(self, images):
        return self.processor(images, return_tensors="pt")["pixel_values"]

    def _get_text_features(self, inputs):
        return self.model.get_text_features(**inputs)

    def _get_image_features(self, inputs):
        return self.model.get_image_features(pixel_values=inputs)

    # def encode_text(self, texts, device):
    #     texts = self.tokenizer(
    #         texts, padding="max_length", return_tensors="pt", truncation=True
    #     ).to(device)
    #     embeddings = self.model.get_text_features(**texts)
    #     embeddings /= embeddings.norm(dim=-1, keepdim=True)
    #     return embeddings

    # def encode_image(self, images, device):
    #     images = self.processor(images, return_tensors="pt").to(device)
    #     embeddings = self.model.get_image_features(**images)
    #     embeddings /= embeddings.norm(dim=-1, keepdim=True)
    #     return embeddings


class OpenCLIP(ContrastiveModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.tag = "OpenCLIP"

    def load_model(self, model_args, device):
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        checkpoint = torch.load(model_args.model_path, map_location="cpu")
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.to(device)
        print(f"Model loaded from: {model_args.model_path}")

    def _prepare_text(self, texts):
        return self.tokenizer(texts)

    def _prepare_image(self, images):
        if not isinstance(images, list):
            images = [images]
        return torch.stack([self.processor(img) for img in images])

    def _get_text_features(self, inputs):
        return self.model.encode_text(inputs)

    def _get_image_features(self, inputs):
        return self.model.encode_image(inputs)

    # def encode_text(self, texts, device):
    #     texts = self.tokenizer(texts).to(device)
    #     embeddings = self.model.encode_text(texts)
    #     embeddings /= embeddings.norm(dim=-1, keepdim=True)
    #     return embeddings

    # def encode_image(self, images, device):
    #     images = [self.processor(image) for image in images]
    #     images = torch.stack(images, dim=0).to(device)
    #     embeddings = self.model.encode_image(images)
    #     embeddings /= embeddings.norm(dim=-1, keepdim=True)
    #     return embeddings
