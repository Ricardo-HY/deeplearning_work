import json
import os.path as osp
from typing import Union
import os


class Prompter(object):
    __slots__ = ("template")

    def __init__(self, template_name: str = ""):

        # 默认模型名称 "alpaca"
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"

        # 加载 JSON 模板
        cur_dir = "/notebook/alpace_lora/templates"
        file_name = osp.join(cur_dir, "alpaca.json")
        # file_name = osp.join(cur_dir, f"{template_name}.json")

        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)


    # 根据提供的指令，输入和标签生成提示
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # 如果有 input，使用存储在 JSON 文件中的模板格式化提示
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        
        # 如果有标签，和将其附加在提示中
        if label:
            res = f"{res}{label}"

        return res

    # 从输出字符串中提取响应
    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()