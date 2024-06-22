from dataclasses import field, dataclass
from transformers import (
    Trainer,
    HfArgumentParser,
    set_seed,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
import os
import torch
import fire
from typing import List

from transformers import LlamaForCausalLM, LlamaTokenizer
from dataset import instruction

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class MyTrainingArguments(TrainingArguments):
    # model / data paramters
    base_model: str = "/root/autodl-tmp/llama-7b-hf"
    data_path: str = "/root/autodl-tmp/alpaca_data.json"
    output_dir: str = "./out/lora-alpace"

    # training paramters
    batch_size: int = 4
    num_epochs: int = 1
    learing_rate: float = 3e-4
    max_len: int = 256
    test_size: int = 1000  # -1 表示：全量
    fp16: bool = True
    logging_steps: int = 10
    optim = "adamw_torch"
    # full_determinism: bool = False
    gradient_accumulation_steps: int = (
        1  # 用于控制在进行参数更新之前要累积的批次数量，以实现梯度累积的效果。
    )
    seed: int = 42
    resume_from_checkpoint: str = None  # either training checkpoint or final adapter
    prompt_template_name: str = (
        "alpace"  # The prompt template to use, will default to alpaca.
    )
    # accelerator_config: str = "default"


@dataclass
class MyFinetuneArguments:

    quantization: str = "8bit"
    device: str = "cuda"

    # lora paramters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


def main():
    training_args, args = HfArgumentParser(
        (MyTrainingArguments, MyFinetuneArguments)
    ).parse_args_into_dataclasses()

    set_seed(training_args.seed)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"world size {world_size} local rank {local_rank}")

    ############# prepare model ##########################

    model = LlamaForCausalLM.from_pretrained(
        training_args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(training_args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type="CAUSAL_LM",
    )

    # 准备模型以进行 int8 的训练
    model = prepare_model_for_kbit_training(model)

    # 使用 PEFT 进行改进模型
    model = get_peft_model(model, config)

    ############# prepare data ##########################

    # 加载构造好的数据集
    data = instruction(
        prompt_tempate_name="alapce",
        tokenizer=tokenizer,
        max_length=training_args.max_len,
        add_eos_token=False,
        data_path=training_args.data_path,
    )



    # 切分训练集和测试集
    if training_args.test_size > 0:
        data = data["train"].train_test_split(
            test_size=training_args.test_size, shuffle=True, seed=42
        )
        train_data = data["train"]
        val_data = data["test"]
    else:
        train_data = data["train"]


    ########  train ################
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()

    model.print_trainable_parameters()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    fire.Fire(main)
