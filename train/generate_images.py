import numpy as np
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from neuraltexture_controlnet import NeuralTextureControlNetModel
from PIL import Image

BATCH_SIZE = 32


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel

    else:
        raise ValueError(f"{model_class} is not supported.")


def init_pipeline(controlnet_dir, device=0):
    text_encoder_cls = import_model_class_from_model_name_or_path(
        "stabilityai/stable-diffusion-2-1-base", None
    )
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="unet",
    )
    weight_dtype = torch.float32
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="vae"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="tokenizer",
        use_fast=False,
    )

    controlnet = NeuralTextureControlNetModel.from_pretrained(
        controlnet_dir, torch_dtype=weight_dtype
    )

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.requires_grad_(False)
    return pipeline


def generate_validation_images(
    pipe,
    val_json_path,
    export_dir,
    num_inference_steps=10,
    cfg_weight=7.5,
    num_batches=10,
):
    from relighting_dataset import RelightingDataset

    validation_dataset = RelightingDataset(
        data_jsonl=val_json_path,
        pretrained_model="stabilityai/stable-diffusion-2-1-base",
        channel_aug_ratio=0,  # add to args
        empty_prompt_ratio=0,  # add to args
        log_encode_hint=False,  # add to args
        load_mask=True,  # add to args
    )

    validation_ref_images = []
    output_file_names = []
    validation_prompts = []
    masks = []
    for i in range(len(validation_dataset)):
        sample = validation_dataset[i]
        validation_prompts.append(sample["text"])
        validation_ref_images.append(
            sample["conditioning_pixel_values"].to(pipe.device)
        )
        source_file_name = sample[
            "ref_file"
        ]  # "dataset_512_v1/eval/obj_60-fabric-green-hedgehog/CF8/013/gt.png"
        target_file_name = sample[
            "target_file"
        ]  # e.g. "dataset_512_v1/eval/obj_60-fabric-green-hedgehog/CF8/010/gt.png"
        object_id = source_file_name.split("/")[2]
        source_light_id = source_file_name.split("/")[-2]
        target_light_id = target_file_name.split("/")[-2]
        view_id = source_file_name.split("/")[-3]
        output_file_name = (
            f"{object_id}_src_{source_light_id}_tgt_{target_light_id}_{view_id}.png"
        )
        mask = sample["mask"]
        masks.append(mask)
        output_file_names.append(output_file_name)

    # Generate images
    # pipe.eval()
    num_generated_batches = 0
    k = 0
    while num_generated_batches < num_batches:
        dir_to_save = os.path.join(export_dir + f"_cfg_{cfg_weight}", f"batch_{k}")
        if os.path.exists(dir_to_save):
            k += 1
            continue
        os.makedirs(dir_to_save, exist_ok=True)
        generated_images = []
        for i in range(0, len(validation_prompts), BATCH_SIZE):
            generated_images.extend(
                pipe(
                    validation_prompts[i : i + BATCH_SIZE],
                    validation_ref_images[i : i + BATCH_SIZE],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg_weight,
                ).images
            )
            torch.cuda.empty_cache()

        for i, image in enumerate(generated_images):
            image = mask_background(image, masks[i])

            # image reshape to 128 if it is 512
            if image.size[0] == 512:
                image = image.resize((128, 128))
            image.save(os.path.join(dir_to_save, output_file_names[i]))
            print(f"Saved image {os.path.join(dir_to_save, output_file_names[i])}")

        del generated_images
        torch.cuda.empty_cache()
        num_generated_batches += 1


def mask_background(img, mask):
    """
    mask background of image

    Args:
        img (PIL): image to be masked, rgba
        mask (np.ndarray): mask to be applied, [0,1] normalized, (w,h)

    Returns:
        img (PIL): masked image
    """
    img = np.array(img)
    # make mask 3 channel
    mask = np.repeat(mask[..., None], 3, axis=2)
    img = np.where(mask, img, 255)
    return Image.fromarray(img)


def main(
    controlnet_dir,
    jsonl_path,
    export_dir,
    device=0,
    num_inference_steps=50,
    guidance_scale=50,
    num_batches=5,
):
    pipeline = init_pipeline(controlnet_dir, device=device)
    generate_validation_images(
        pipeline,
        jsonl_path,
        export_dir,
        num_inference_steps=num_inference_steps,
        cfg_weight=guidance_scale,
        num_batches=num_batches,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

    # example command
    # python generate_images.py \
    #  --controlnet_dir /path/to/controlnet \
    # --jsonl_path /path/to/val.jsonl \
    #  --export_dir /path/to/export_dir\
    #  --device 0
