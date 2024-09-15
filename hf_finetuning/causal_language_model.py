import logging
import os
import sys
from typing import Optional, Any, Dict, Tuple,List
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING
)
import transformers
from transformers.utils import send_example_telemetry, logging as hf_logging
from transformers.trainer_utils import get_last_checkpoint
from transformers.testing_utils import CaptureLogger
from hfconfig import DataTrainingArguments, ModelArguments

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
logger.info(f"\nModel types: {MODEL_TYPES}")

def load_yaml_config(yaml_file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(yaml_file_path, "r", encoding="utf-8") as yml_file:
        return yaml.safe_load(yml_file)

def setup_logging(training_args: TrainingArguments) -> None:
    """Setup logging for the training process."""
    if training_args.should_log:
        hf_logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def detect_last_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    """Detect the last checkpoint if resuming training."""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overwrite the existing files."
            )
    return last_checkpoint

def configure_model_and_tokenizer(
    model_args: ModelArguments
) -> Tuple[AutoConfig, AutoTokenizer, Optional[AutoModelForCausalLM]]:
    """Configure the model and tokenizer from the provided arguments."""
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "Please use --tokenizer_name to specify a pre-trained tokenizer."
        )

    if model_args.model_name_or_path:
        torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype not in ["auto", None] else None
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    return config, tokenizer, model

def tokenize_function(
    examples: Dict[str, Any],
    input_column_names: List[str],
    target_column_name: str,
    tokenizer: Any,
    tok_logger: logging.Logger,
    max_length: int=512
) -> Dict[str, Any]:
    """
    Tokenize the input and target columns in the dataset.
   
    Args:
        examples: A dictionary containing the data batch.
        input_column_names: A list of column names to be used as input.
        target_column_name: The target column name for labels.
        tokenizer: The tokenizer to be used for tokenization.
        tok_logger: Logger to capture tokenizer warnings or output.
        max_length: Maximum length for tokenized sequences.
   
    Returns:
        A dictionary with tokenized input and labels.
    """
    with CaptureLogger(tok_logger) as cl:
        # Concatenate input columns to form the input text for each example in the batch
        input_texts = [" ".join(str(examples[col][i]) for col in input_column_names) for i in range(len(examples[input_column_names[0]]))]
        target_texts = [str(examples[target_column_name][i]) for i in range(len(examples[target_column_name]))]
        
        model_inputs = tokenizer(input_texts, add_special_tokens=True, padding=False, truncation=False)
        labels = tokenizer(target_texts, add_special_tokens=False, padding=False, truncation=False)

        for i in range(len(input_texts)):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

            combined_input_ids = sample_input_ids + label_input_ids
            combined_labels = [-100] * len(sample_input_ids) + label_input_ids

            padding_length = max(0, max_length - len(combined_input_ids))

            # Pad input_ids, attention_mask, and labels
            padded_input_ids = [tokenizer.pad_token_id] * padding_length + combined_input_ids
            padded_attention_mask = [0] * padding_length + [1] * len(combined_input_ids)
            padded_labels = [-100] * padding_length + combined_labels

            # Truncate to ensure no overflow beyond max_length
            model_inputs["input_ids"][i] = torch.tensor(padded_input_ids[:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(padded_attention_mask[:max_length])
            labels["input_ids"][i] = torch.tensor(padded_labels[:max_length])

        model_inputs["labels"] = labels["input_ids"]

    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "Please ignore the warning above - this long input will be chunked "
            "into smaller bits before being passed to the model."
        )

    return model_inputs



def main(yaml_file_path: str):
    # Load YAML configuration
    config = load_yaml_config(yaml_file_path=yaml_file_path)
    
    # Mapping YAML values to dataclasses
    model_args = ModelArguments(**config['ModelArguments'])
    data_args = DataTrainingArguments(**config["DataTrainingArguments"])
    training_args = TrainingArguments(**config["TrainingArguments"])
    
    # Send telemetry
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    setup_logging(training_args)
    

    # Log the process summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: "
        f"{training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    # Detect last checkpoint
    last_checkpoint = detect_last_checkpoint(training_args)
    if last_checkpoint:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Download the dataset
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets.keys():
            split_percentage = data_args.validation_split_percentage
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file:
            data_files["train"] = data_args.train_file
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        if "validation" not in raw_datasets.keys():
            split_percentage = data_args.validation_split_percentage
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    # Configure model, tokenizer, and their configurations
    config, tokenizer, model = configure_model_and_tokenizer(model_args)

    logger.info(f"\nModel config:{config}")

    # Resize token embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    # Preprocessing the datasets
    column_names = raw_datasets["train"].column_names
    input_column_names = data_args.input_column_names if data_args.input_column_names else column_names
    target_column_name = data_args.target_column_name
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            lambda examples: tokenize_function(examples, input_column_names, target_column_name, tokenizer, tok_logger,max_length=data_args.max_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    block_size = min(data_args.block_size, tokenizer.model_max_length)
    logger.info(f"Tokenizing into block size: {block_size}")

    def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Group texts into chunks of block size."""
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    with training_args.main_process_first(desc="Grouped text tokenization"):
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Grouping texts into blocks",
        )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy loading

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(tokenized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(tokenized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(tokenized_datasets["validation"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_datasets["validation"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Training completed.")

if __name__ == "__main__":
    try:
        yaml_config_file = sys.argv[1]
    except IndexError:
        raise ValueError("Please provide the path to the YAML configuration file as the first argument.")

    main(yaml_config_file)