本地设置
安装依赖项
pip install -r requirements.txt
如果比特字节不起作用，从源代码安装它

培训(finetune.py)
该文件包含PEFT对LLaMA模型的直接应用程序，以及一些与提示构造和标记化相关的代码。
示例用法：
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
--output_dir './lora-alpaca'
我们还可以调整超参数：
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
--group_by_length

推断(generate.py)
此文件从Hugging Face模型中心读取基础模型，并从中读取LoRA权重tloen/alpaca-lora-7b，并运行Gradio接口以对指定的输入进行推理。用户应将其视为使用模型的示例代码，并根据需要进行修改。
示例用法：
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
--lora_weights 'tloen/alpaca-lora-7b'

检查点导出(export_*_checkpoint.py )
这些文件包含将LoRA权重合并回基本模型的脚本，以便导出为Hugging Face格式和PyTorchstate_dicts。它们应该帮助希望在以下项目中运行推断的用户呼叫.cpp或alpaca.cpp 
.
Docker设置和推断
生成容器映像：
docker build -t alpaca-lora .
运行容器（也可以使用finetune.py以及上面所示的用于培训的所有参数）：
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'

Docker编写设置和推断
（可选）更改所需型号和重量environment
构建并运行容器
docker-compose up -d --build
参阅日志：
docker-compose logs -f
清理一切：
docker-compose down --volumes --rmi all

笔记
如果有更好的数据集，可能会显著提高模型性能。考虑支持LAION开放式助理产生一个高质量的数据集，用于监督微调。
