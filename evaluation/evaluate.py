from accelerate import Accelerator
from accelerate.utils import gather_object
from dataclasses import dataclass, field, asdict
import json
from typing import Optional
from transformers import HfArgumentParser

from models import HuggingFaceCLIP, OpenCLIP


@dataclass
class ModelArguments:
    model_variant: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace/OpenCLIP"}
    )
    model_name_or_path: Optional[str] = field(
        default="openai/clip-vit-large-patch14", metadata={"help": "CLIP Model"}
    )
    tokenizer_name: Optional[str] = field(
        default="", metadata={"help": "Tokenizer used to compute similarity"}
    )
    image_processor_name: Optional[str] = field(
        default="", metadata={"help": "Image Processor used to compute similarity"}
    )
    local_files_only: Optional[bool] = field(
        default=False, metadata={"help": "If using only local files"}
    )
    lora: Optional[bool] = field(
        default=False, metadata={"help": "Load lora model using with Peft."}
    )
    output_dir: Optional[str] = field(
        default="", metadata={"help": "Path to store results"}
    )


def main():
    parser = HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]
    accelerator = Accelerator()
    device = accelerator.device

    if model_args.model_variant == "HuggingFace":
        model = HuggingFaceCLIP()
    elif model_args.model_variant == "OpenCLIP":
        model = OpenCLIP()
    else:
        raise ValueError(
            f"Invalid model_variant: {model_args.model_variant}. "
            "Must be 'HuggingFace' or 'OpenCLIP'."
        )

    model.load_model(model_args, accelerator.process_index)

    import code

    code.interact(local=locals())

    # save parameters
    if False:
        with open("model_args.json", "w") as f:
            json.dump(asdict(model_args), f, indent=4)


if __name__ == "__main__":
    main()
