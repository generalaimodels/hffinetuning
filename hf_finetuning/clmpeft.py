import os
from typing import Dict, Any, List

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from peft import (
    get_peft_config,
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    PeftConfig
)
from datasets import load_dataset
from tqdm import tqdm
import logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Error loading configuration: {e}")


def configure_logging() -> logging.Logger:
    """Configure and return a logger."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def preprocess_function(
    examples: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    text_column: str,
    label_column: str,
    max_length: int
) -> Dict[str, torch.Tensor]:
    """Preprocess the dataset for training."""
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    labels = tokenizer(targets, add_special_tokens=False, max_length=max_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding token id's of the labels by -100
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in model_inputs["labels"]
    ]

    # Convert to tensors
    model_inputs = {k: torch.tensor(v) for k, v in model_inputs.items()}

    return model_inputs


def test_preprocess_function(
    examples: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    text_column: str,
    max_length: int
) -> Dict[str, torch.Tensor]:
    """Preprocess the dataset for testing."""
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)

    # Convert to tensors
    model_inputs = {k: torch.tensor(v) for k, v in model_inputs.items()}

    return model_inputs

def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    device: torch.device,
    max_length: int
) -> str:
    """Run inference on a single input text."""
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output.split("Label :")[-1].strip()

def run_tests(
    config: Dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    logger: logging.Logger
) -> None:
    """Run tests on the model and perform inference."""
    try:
        dataset_name = config['dataset_name']
        text_column = config['text_column']
        max_length = config['max_length']
        device = torch.device(config['device'])

        logger.info("Loading test dataset...")
        dataset = load_dataset("ought/raft", dataset_name)
        test_dataset = dataset["test"]

        correct_predictions = 0
        total_predictions = 0

        for example in tqdm(test_dataset, desc="Testing and Inferencing"):
            input_text = f"{text_column} : {example[text_column]} Label : "
            true_label = example['Label']
            
            predicted_label = run_inference(model, tokenizer, input_text, device, max_length)
            
            logger.info(f"Input: {example[text_column]}")
            logger.info(f"True Label: {true_label}")
            logger.info(f"Predicted Label: {predicted_label}")
            logger.info("---")

            if predicted_label == true_label:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions
        logger.info(f"Test Accuracy: {accuracy:.4f}")

    except Exception as error:
        logger.error(f"An error occurred during testing and inference: {error}")
        raise

def main() -> None:
    """Main function to run the script."""
    config_path = 'config.yml'
    config = load_config(config_path)

    logger = configure_logging()

    try:
        device = torch.device(config['device'])
        model_name_or_path = config['model_name_or_path']
        tokenizer_name_or_path = config['tokenizer_name_or_path']
        dataset_name = config['dataset_name']
        num_virtual_tokens = config['num_virtual_tokens']
        prompt_tuning_init_text = config['prompt_tuning_init_text']
        text_column = config['text_column']
        label_column = config['label_column']
        max_length = config['max_length']
        learning_rate = config['learning_rate']
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=model_name_or_path,
        )

        logger.info("Loading dataset...")
        dataset = load_dataset("ought/raft", dataset_name)

        classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
        logger.info(f"Classes: {classes}")

        dataset = dataset.map(
            lambda x: {label_column: [classes[label] for label in x["Label"]]},
            batched=True,
            num_proc=os.cpu_count(),
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        processed_datasets = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, text_column, label_column, max_length),
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["train"]

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(torch.tensor(train_epoch_loss))
            logger.info(f"Epoch {epoch+1} Train Loss: {train_epoch_loss:.4f} PPL: {train_ppl:.4f}")

            model.eval()
            eval_loss = 0.0
            for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}")):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(torch.tensor(eval_epoch_loss))
            logger.info(f"Epoch {epoch+1} Eval Loss: {eval_epoch_loss:.4f} PPL: {eval_ppl:.4f}")

               # Save the model
        model.save_pretrained(f"peft_model_{dataset_name}")
        
        # Load the saved model for inference
        peft_model_path = f"peft_model_{dataset_name}"
        config_path = os.path.join(peft_model_path, "adapter_config.json")
        model_path = os.path.join(peft_model_path, "adapter_model.bin")
        
        if os.path.exists(config_path) and os.path.exists(model_path):
            # Load the base model
            base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            
            # Load the PEFT configuration
            peft_config = PeftConfig.from_pretrained(peft_model_path)
            
            # Get the PEFT model
            model = get_peft_model(base_model, peft_config)
            
            # Load the trained weights
            model.load_state_dict(torch.load(model_path))
            
            model.to(device)
            model.eval()
            
            # Run tests
            run_tests(config, model, tokenizer, logger)
        else:
            logger.error(f"Saved model not found at {peft_model_path}")
        
    except Exception as error:
        logger.error(f"An error occurred: {error}")
        raise


if __name__ == "__main__":
    main()