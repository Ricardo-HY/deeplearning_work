from utils.prompter import Prompter
from transformers import LlamaTokenizer

from datasets import load_dataset


def instruction(tokenizer, **args):
    prompter = Prompter(template_name=args["prompt_tempate_name"])
    data_path = args.get("data_path", None)
    max_length = args.get("max_length", None)

    

    # 用于标记化文本数据
    def tokenize(prompt, add_eos_token=True, **args):

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        # 若"input_ids"结尾不是<EOS>且长度小于最大值且有<EOS>，添加<EOS>
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)


        # 找了一下午
        result["labels"] = result["input_ids"].copy()

        return result

    # 用于生成和标记化提示
    def generate_and_tokenize_prompt(data_point):

        # 根据提供的指令，输入和标签生成提示
        full_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"], data_point["output"]
        )

        # 将生成提示标记化
        tokenized_full_prompt = tokenize(full_prompt, **args)


        return tokenized_full_prompt

    data = load_dataset(
        "json", data_files=data_path
    )

    # 将"generate_and_tokenize_prompt"用于数据中每个数据点，并进程并行处理（一条一条处理）
    data = data.map(generate_and_tokenize_prompt, num_proc=8)
    return data


if __name__ == "__main__":

    tokenizer = LlamaTokenizer.from_pretrained(
        "D://Downloads//llm//LLaMA-7B//tokenizer.model"
    )

    ds = instruction(
        prompt_tempate_name="alpaca",
        tokenizer=tokenizer,
        max_length=1024,
        add_eos_token=False,
    )

    print(ds['train'][1])
    print(tokenizer.decode(ds['train'][1]['input_ids']))
