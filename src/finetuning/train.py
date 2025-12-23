from pathlib import Path
from typing import Optional
import json


def finetune_model(
    corpus_path: str,
    output_dir: str = "./models/finetuned",
    max_steps: int = 50,
    max_seq_length: int = 2048
) -> Path:
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        import torch
    except ImportError:
        raise ImportError("unsloth library required. Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
    
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
        poems = corpus_data.get("poems", [])
    
    if not poems:
        raise ValueError("No poems found in corpus")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    train_data = [{"instruction": "Écris un poème en français.", "output": poem} for poem in poems]
    
    from datasets import Dataset
    dataset = Dataset.from_list(train_data)
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|><|end_of_text|>" for inst, out in zip(instructions, outputs)]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir=output_dir,
            optim="adamw_8bit",
            seed=3407,
        ),
    )
    
    trainer.train()
    
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    training_history_file = results_dir / "training_history.json"
    
    training_history = trainer.state.log_history
    with open(training_history_file, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    return output_path

