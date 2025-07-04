from accelerate import Accelerator
from dataclasses import dataclass, field, asdict
import json
from typing import Optional, List
from transformers import HfArgumentParser

from evaluation.tasks.winoground import evaluate_winoground
from models import HuggingFaceCLIP, OpenCLIP, LongCLIP
from pathlib import Path
from tasks import (
    evaluate_classification,
    evaluate_retrieval,
    evaluate_aro,
    evaluate_scpp,
)


@dataclass
class ModelArguments:
    model_variant: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace/OpenCLIP"}
    )
    model_path: Optional[str] = field(default=None, metadata={"help": "Model path"})
    processor_path: Optional[str] = field(
        default=None, metadata={"help": "Processor path if different from model"}
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


@dataclass
class DataArguments:
    aro: Optional[bool] = field(
        default=False, metadata={"help": "Evaluate ARO datasets."}
    )
    scpp: Optional[bool] = field(
        default=False, metadata={"help": "Evaluate scpp dataset."}
    )
    classification: Optional[bool] = field(
        default=False, metadata={"help": "Evaluate classification."}
    )
    retrieval: Optional[List[str]] = field(
        default_factory=list,  # Default empty list
        metadata={"help": "List of dataset names for retrieval evaluation."},
    )


def main():
    parser = HfArgumentParser([ModelArguments, DataArguments])
    model_args, data_args = parser.parse_args_into_dataclasses()
    accelerator = Accelerator()
    device = accelerator.device

    if model_args.model_variant == "HuggingFace":
        model = HuggingFaceCLIP()
    elif model_args.model_variant == "OpenCLIP":
        model = OpenCLIP()
    elif model_args.model_variant == "LongCLIP":
        model = LongCLIP()
    else:
        raise ValueError(
            f"Invalid model_variant: {model_args.model_variant}. "
            "Must be 'HuggingFace' or 'OpenCLIP'."
        )

    model.load_model(
        model_path=model_args.model_path,
        device=accelerator.process_index,
        processor_path=model_args.processor_path,
        lora=model_args.lora,
    )
    scores = {"model": model_args.model_path, "experiments": {}}
    if data_args.classification:
        scores["experiments"]["classification"] = evaluate_classification(model, device)
    if data_args.retrieval:
        scores["experiments"]["retrieval"] = evaluate_retrieval(
            data_args.retrieval, model, device
        )
    if data_args.aro:
        scores["experiments"]["aro"] = evaluate_aro(model, device)
    if data_args.scpp:
        scores["experiments"]["scpp"] = evaluate_scpp(model, device)
    if data_args.winoground:
        scores["experiments"]["winoground"] = evaluate_winoground(model, device)
    # save results
    if not model_args.output_dir:
        model_name = Path(model_args.model_path).stem
        model_name.mkdir(parents=True, exist_ok=True)
        model_args.output_dir = f".results/{model_name}/results.json"
    Path(model_args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    with open(model_args.output_dir, "w") as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    main()
