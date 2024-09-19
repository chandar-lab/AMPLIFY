import os
import numpy as np
from argparse import ArgumentParser, Action
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, EsmForMaskedLM, TrainerCallback
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


if __name__ == "__main__":
    parser = ArgumentParser()
    # Model
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--tokenizer_path", type=str, help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--resume", action="store_true", help="")
    # Dataset
    parser.add_argument("--train_paths", type=str, nargs="*", help="")
    parser.add_argument("--streaming", action="store_true", help="")
    parser.add_argument(
        "--eval_dict",
        action=type(
            "", (Action,), dict(__call__=lambda a, p, n, v, o: getattr(n, a.dest).update(dict([v.split("=")])))
        ),
        default={},
    )
    # Log
    parser.add_argument("--run_name", type=str, help="")
    parser.add_argument("--output_dir", type=str, help="")
    args = parser.parse_args()

    # Initialize wandb manually for more control
    os.environ["WANDB_ENTITY"] = "drug-discovery"
    os.environ["WANDB_PROJECT"] = "protein-scorer"
    os.environ["WANDB_DIR"] = os.path.join(args.output_dir, args.run_name)
    os.environ["WANDB_MODE"] = "online"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def tokenize(x):
        return tokenizer(x["sequence"], truncation=True, max_length=512, padding="max_length")

    data_train = load_dataset("csv", data_files=args.train_paths, split="train", streaming=args.streaming)
    data_train = data_train.shuffle(seed=0).map(tokenize, num_proc=32, batched=True)

    data_eval = dict()
    for name, path in args.eval_dict.items():
        data_eval[name] = load_dataset("csv", data_files=path, split="train").map(tokenize, num_proc=32, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    model = EsmForMaskedLM(AutoConfig.from_pretrained(args.model_path))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        num_pred = np.sum(labels != -100).item()
        num_correct = np.sum(np.argmax(logits, axis=-1) == labels).item()
        return {"accuracy": num_correct / num_pred}

    trainer_args = TrainingArguments(
        evaluation_strategy="steps",
        output_dir=os.path.join(args.output_dir, args.run_name),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=4e-4,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        warmup_steps=2000,
        max_steps=500000,
        save_total_limit=10,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=8,
        bf16=True,
        tf32=True,
        max_grad_norm=None,
        report_to="wandb",
        run_name=args.run_name,
        torch_compile=True,
        logging_steps=100,
        eval_steps=1000,
        save_steps=1000,
    )

    class WeightNormCallback(TrainerCallback):
        def on_log(self, args, state, control, model, **kwargs):
            kwargs["logs"]["weight_norm"] = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5

    trainer = Trainer(
        model,
        trainer_args,
        data_collator,
        data_train,
        data_eval,
        callbacks=[WeightNormCallback()],
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume)
