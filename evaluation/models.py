from abc import ABC, abstractmethod
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from peft import PeftModel
import open_clip
from longclip.model import longclip


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
    def load_model(self, model_path, device, processor_path=None, lora=None, **kwargs):
        pass

    def encode_text(self, texts, device):
        texts = self._prepare_text(texts)
        embeddings = self._get_text_features(texts.to(device))
        return self._normalize(embeddings)

    def encode_image(self, images, device, prepare=True):
        if prepare:
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

    def load_model(self, model_path, device, processor_path=None, lora=None):
        if lora:
            base = CLIPModel.from_pretrained(
                BASE,
                local_files_only=True,
            )
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model.merge_and_unload()
        else:
            self.model = CLIPModel.from_pretrained(
                model_path,
                local_files_only=True,
            )
        print(f"Model loaded from: {model_path}")
        if processor_path is None:
            processor_path = model_path

        self.tokenizer = CLIPTokenizer.from_pretrained(
            processor_path, local_files_only=True
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            processor_path, local_files_only=True
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


class OpenCLIP(ContrastiveModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.tag = "OpenCLIP"

    def load_model(self, model_path, device, processor_path=None, lora=None):
        if lora:
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=device,
                lora=4,
            )
        else:
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=device
            )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        checkpoint = torch.load(model_path, map_location="cpu")
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.to(device)
        print(f"Model loaded from: {model_path}")

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


class LongCLIP(ContrastiveModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.tag = "longclip"

    def load_model(self, model_path, device, processor_path=None, lora=None, **kwargs):
        self.model, self.processor = longclip.load(model_path, device=device)

    def _prepare_text(self, texts):
        return longclip.tokenize(texts, truncate=True)

    def _prepare_image(self, images):
        if not isinstance(images, list):
            images = [images]
        return torch.stack([self.processor(img) for img in images])

    def _get_text_features(self, inputs):
        return self.model.encode_text(inputs)

    def _get_image_features(self, inputs):
        return self.model.encode_image(inputs)
