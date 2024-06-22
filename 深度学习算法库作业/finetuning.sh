torchrun --nproc_per_node 1 finetuning.py  \
    --base_model /root/autodl-tmp/llama-7b-hf  \
    --data_path /root/autodl-tmp/alpaca_data.json \
    # --output_dir ./out/lora-alpace \
    # --batch_size 4 \
    # --num_epochs 1 \
    # --learing_rate 3e-4 \
    # --max_len 256 \
    # --test_size 1000 \
    # --fp16 True \
    # --logging_steps 10 \
    # --optim adamw_torch \
    # --gradient_accumulation_steps 1 \
    # --seed 42 \
    # --prompt_template_name alpace \