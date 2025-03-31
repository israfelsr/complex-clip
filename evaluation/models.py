from abc import ABC, abstractmethod
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from peft import PeftModel
import open_clip

MODEL_ROOT = "/leonardo_work/EUHPC_D12_071/data/HF/clip-base"


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


class HuggingFaceCLIP(ContrastiveModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.tag = "huggingface"

    def load_model(self, model_args, device):
        if model_args.lora:
            base = CLIPModel.from_pretrained(
                MODEL_ROOT,
                local_files_only=True,
            )
            self.model = PeftModel.from_pretrained(base, model_args.model_path)
        else:
            self.model = CLIPModel.from_pretrained(
                model_args.model_path,
                local_files_only=model_args.local_files_only,
            )
        print(f"Model loaded from: {model_args.model_path}")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_args.model_path, local_files_only=model_args.local_files_only
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            model_args.model_path,
            local_files_only=model_args.local_files_only,
        )
        self.model.to(device)

    def encode_text(self, texts, device):
        texts = self.tokenizer(
            texts, padding="max_length", return_tensors="pt", truncation=True
        ).to(device)
        embeddings = self.model.get_text_features(**texts)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def encode_image(self, images, device):
        images = self.processor(images, return_tensors="pt").to(device)
        embeddings = self.model.get_image_features(**images)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings


class OpenCLIP(ContrastiveModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.tag = "openclip"

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

    def encode_text(self, texts, device):
        texts = self.tokenizer(texts).to(device)
        embeddings = self.model.encode_text(texts)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def encode_image(self, images, device):
        images = [self.processor(image) for image in images]
        images = torch.stack(images, dim=0).to(device)
        embeddings = self.model.encode_image(images)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings
