import logging
import os
import math
import sys
from typing import Optional, Any, Dict, Tuple,List
import torch
import yaml
import json
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    default_data_collator,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    get_scheduler,
)
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm
from torch.utils.data import DataLoader
import transformers
from transformers.utils import send_example_telemetry, logging as hf_logging
from transformers.trainer_utils import get_last_checkpoint
from transformers.testing_utils import CaptureLogger
from hfconfig import DataTrainingArguments, ModelArguments, TrainingArgumentsAcc
from accelerate import Accelerator, DistributedType
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
logger.info(f"Model types: {MODEL_TYPES}")

def load_yaml_config(yaml_file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(yaml_file_path, "r", encoding="utf-8") as yml_file:
        return yaml.safe_load(yml_file)

def setup_logging(training_args: TrainingArguments) -> None:
    """Setup logging for the training process."""
    if training_args.should_log:
        hf_logging.set_verbosity_info()
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # transformers.utils.logging.set_verbosity()
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

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
    training_args = TrainingArgumentsAcc(**config["TrainingArguments"])
    print(training_args,data_args,model_args)
    # Send telemetry
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    setup_logging(training_args)
    
    # Log the process summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: "
        # f"{training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = detect_last_checkpoint(training_args)
    if last_checkpoint:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if training_args.with_tracking:
        accelerator_log_kwargs["log_with"] = training_args.report_to
        accelerator_log_kwargs["project_dir"] = training_args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps, **accelerator_log_kwargs)
    logger.info(accelerator.state, main_process_only=False)
    # Set seed before initializing model
    set_seed(training_args.seed)
      # Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
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

    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Grouping texts into blocks",
    )
    
    
     # DataLoaders creation:
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=default_data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], collate_fn=default_data_collator, batch_size=training_args.per_device_eval_batch_size
    )
    
     # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=training_args.max_train_steps
        if overrode_max_train_steps
        else training_args.max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
     # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)
    
    
     # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = training_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    
    
     # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if training_args.with_tracking:
        experiment_config = vars(training_args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    
    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            checkpoint_path = training_args.resume_from_checkpoint
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        if training_args.with_tracking:
            total_loss = 0
        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if training_args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= training_args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if training_args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if training_args.push_to_hub and epoch < training_args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(training_args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=training_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=training_args.hub_token,
                )

        if training_args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if training_args.output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if training_args.with_tracking:
        accelerator.end_training()

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=training_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=training_args.hub_token,
                )
            with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

    logger.info("Training completed.")

if __name__ == "__main__":
    try:
        yaml_config_file = sys.argv[1]
    except IndexError:
        raise ValueError("Please provide the path to the YAML configuration file as the first argument.")

    main(yaml_config_file)