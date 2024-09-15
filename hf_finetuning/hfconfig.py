from typing import Optional, Union, Sequence, Mapping, Dict, Any,List
from dataclasses import dataclass,field


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None)
    low_cpu_mem_usage: bool = field(default=False)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    streaming: bool = field(default=False)
    block_size: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    keep_linebreaks: bool = field(default=True)
    input_column_names: List[str] = field(default_factory=list)  # List of input column names
    target_column_name: Optional[str] = field(default=None)       # Name of the target column for labels
    max_length:Optional[str]=field(default=int(512))

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json, or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json, or a txt file."

        if not self.input_column_names:
            raise ValueError("`input_column_names` must be a non-empty list specifying the input columns.")
        if self.target_column_name is None:
            raise ValueError("`target_column_name` must be specified.")


@dataclass
class TrainingArgumentsAcc:
    output_dir: str = field(metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    overwrite_output_dir: bool = field(default=True, metadata={"help": "Overwrite the content of the output directory if it exists."})
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation on the validation set."})
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Batch size per device during training."})
    per_device_eval_batch_size: int = field(default=16, metadata={"help": "Batch size per device during evaluation."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay to apply (if any)."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_train_steps: Optional[int] = field(default=None, metadata={"help": "If set, overrides num_train_epochs to train up to max_train_steps steps."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use."})
    num_warmup_steps: int = field(default=500, metadata={"help": "Linear warmup over warmup_steps."})
    logging_dir: str = field(default="./logs", metadata={"help": "Tensorboard log directory."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=5000, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: int = field(default=2, metadata={"help": "Limit the total amount of checkpoints."})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    fp16: bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision."})
    evaluation_strategy: str = field(default="epoch", metadata={"help": "Evaluation strategy to use during training."})
    save_strategy: str = field(default="epoch", metadata={"help": "Save strategy to use during training."})
    push_to_hub: bool = field(default=False, metadata={"help": "Whether to push the model to the Hugging Face Hub."})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "Model id for the Hugging Face Hub."})
    hub_token: Optional[str] = field(default=None, metadata={"help": "Token for pushing to the Hugging Face Hub."})
    report_to: str = field(default="tensorboard", metadata={"help": "The list of integrations to report the results and logs to."})
    checkpointing_steps: str = field(default="epoch", metadata={"help": "Save checkpoints at each epoch."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to a checkpoint folder to resume training."})
    with_tracking: bool = field(default=True, metadata={"help": "Whether to enable experiment tracking."})
    should_log: bool = field(default=True, metadata={"help": "Whether to log the training steps."})  # Newly added attribute
    get_process_log_level: bool = field(default=True, metadata={"help": "Whether to log the training steps."}) 
    local_rank : int =field(default=0)
    device :Optional[str] =field(default="cpu")
    n_gpu :Optional[float] =field(default=0)
    parallel_mode :Optional[int]=field(default=2)