from accelerate import Accelerator
from accelerate.utils import gather_object
from dataclasses import dataclass, field, asdict
import json
from typing import Optional
from transformers import HfArgumentParser

from models import HuggingFaceCLIP, OpenCLIP
from tasks.classification import evaluate_classification


@dataclass
class ModelArguments:
    model_variant: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace/OpenCLIP"}
    )
    model_path: Optional[str] = field(default="", metadata={"help": "Model path"})
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
    classification: Optional[bool] = field(
        default=False, metadata={"help": "Evaluate classification."}
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
    if model_args.classification:
        scores = evaluate_classification(model, device, accelerator)

    import code

    code.interact(local=locals())

    # save parameters
    if False:
        with open("model_args.json", "w") as f:
            json.dump(asdict(model_args), f, indent=4)


if __name__ == "__main__":
    main()
