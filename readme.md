### Train model

- dataset prepare
    - https://drive.google.com/file/d/14w5OHPnh_py73tOeXQItTXOsFmC1eiXP/view 에서 resolution 가공되고 radiance hint가 미리 generation되고 directory 구조가 맞게된 preprocessed dataset을 다운로드 받아서 사용한다.
    - 받아진 dataset을 `train/dataset_v2`에 위치시킨다.

- dependencies
    - conda 환경과 필요한 pip library를 다음과 같이 설정한다.
    ```bash
    conda create --name dilightnet python=3.10 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    conda activate dilightnet
    pip install -r requirements.txt
    ```

- train shell command
    - train
    
    ```python
    cd ./train
    accelerate launch train_controlnet.py \
      --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
      --output_dir=runs/dilightnet-openillum-2-1-2-base-v2 \
      --dataset_name="dataset_v2/train_v2.jsonl" \
      --validation_dataset_name="dataset_v2/eval_v2.jsonl" \
      --resolution=128 \
      --shading_hint_channels=12 \
      --learning_rate=1e-5 \
      --lr_scheduler="linear" \
      --train_batch_size=256 \
      --dataloader_num_workers=8 \
      --report_to=wandb \
      --checkpointing_steps=1000 \
      --validation_steps=250 \
      --max_train_steps=50000 \
      --proportion_empty_prompts=0.2 \
      --proportion_channel_aug=0 \
      --gradient_checkpointing \
      --gradient_accumulation_steps=1 \
      --set_grads_to_none \
      --allow_tf32 \
      --num_validation_images=4 \
      --add_mask \
      --checkpoints_total_limit=5 \
      --tracker_project_name="dilightnet-openillum-main-exp_test" \
    ```
    

### Evaluate model

- model Checkpoint
    - https://drive.google.com/file/d/1R68I3cGs5S8_H7HfNN_X0Wqio1QceCNA/view?usp=sharing
- eval result reproduce shell commnand
    
    ```python
    python generate_images.py \
    --controlnet_dir 'runs/dilightnet-openillum-2-1-2-base-v2.2/checkpoint-45000/controlnet' \
    --jsonl_path 'dataset_v2/eval_v2.jsonl' \
    --export_dir 'generated_images/generated_128_4ksteps' \
    --num_inference_steps 300 \
    --guidance_scale 1.0 \
    --num_batches 20 \
    --device 1 &
    ```