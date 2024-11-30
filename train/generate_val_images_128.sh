python generate_images.py \
--controlnet_dir 'runs/checkpoint-45000/controlnet' \
--jsonl_path 'dataset_v2/eval_v2.jsonl' \
--export_dir 'generated_images/generated_128_4ksteps' \
--num_inference_steps 300 \
--guidance_scale 0.9 \
--num_batches 20 \
--device 0

