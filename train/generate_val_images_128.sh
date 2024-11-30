python generate_images.py \
--controlnet_dir '/workspace/diffusion-project/diffusion_submission/train/runs/dilightnet-openillum-2-1-2-base-v2.2/checkpoint-40000/controlnet' \
--jsonl_path 'dataset_v2/eval_v2.jsonl' \
--export_dir 'generated_images/generated_128_4ksteps' \
--num_inference_steps 300 \
--guidance_scale 0.9 \
--num_batches 20 \
--device 0

