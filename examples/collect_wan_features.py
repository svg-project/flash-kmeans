"""Collect post-RoPE query vectors during independent Wan video generation."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from diffusers import WanPipeline


PROMPTS = [
    "A red fox walks through a snowy forest. The camera follows slowly as snow falls.",
    "A busy harbor at sunset, with fishing boats rocking on the water and birds flying overhead.",
    "A potter shapes a clay bowl on a spinning wheel, close-up view of the hands and wet clay.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--output", type=Path, default=Path("features/wan"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--start-prompt", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to("npu")
    pipe.transformer.set_attention_backend("_native_npu")
    selected_steps = {0, args.steps // 2, args.steps - 1}
    state = {"step": 0, "prompt": 0}
    rotary = {}
    written = set()
    heads = [0, 3, 6, 9]
    handles = []

    for layer in (0, 12, 24):
        attention = pipe.transformer.blocks[layer].attn1

        def before(module, inputs, layer=layer):
            rotary[layer] = inputs[3]

        def capture(module, inputs, output, layer=layer):
            key = (state["prompt"], state["step"], layer)
            if state["step"] not in selected_steps or key in written:
                return
            query = output.unflatten(2, (12, 128))
            cos, sin = rotary[layer]
            first, second = query.unflatten(-1, (-1, 2)).unbind(-1)
            rotated = torch.empty_like(query)
            rotated[..., 0::2] = first * cos[..., 0::2] - second * sin[..., 1::2]
            rotated[..., 1::2] = first * sin[..., 1::2] + second * cos[..., 0::2]
            features = rotated[0, :, heads].permute(1, 0, 2).contiguous().cpu()
            path = args.output / f"prompt{key[0]}_step{key[1]}_layer{layer}.pt"
            torch.save({"features": features, "prompt": PROMPTS[key[0]], "seed": 3100 + key[0],
                        "step": key[1], "layer": layer, "heads": heads,
                        "representation": "post_rope_query"}, path)
            written.add(key)
            print(path, list(features.shape), flush=True)

        handles.append(attention.register_forward_pre_hook(before))
        handles.append(attention.norm_q.register_forward_hook(capture))

    def next_step(pipeline, step, timestep, callback_kwargs):
        state["step"] = step + 1
        return callback_kwargs

    for index in range(args.start_prompt, min(len(PROMPTS), args.start_prompt + args.prompts)):
        prompt = PROMPTS[index]
        state.update(step=0, prompt=index)
        result = pipe(prompt=prompt, height=args.height, width=args.width, num_frames=args.frames,
                      num_inference_steps=args.steps, guidance_scale=5.0,
                      generator=torch.Generator(device="cpu").manual_seed(3100 + index),
                      callback_on_step_end=next_step, output_type="latent")
        torch.save({"latents": result.frames.cpu(), "prompt": prompt}, args.output / f"prompt{index}_latents.pt")
        config = vars(args).copy()
        config["output"] = str(args.output)
        config.update(prompt=prompt, prompt_index=index, seed=3100 + index,
                      layers=[0, 12, 24], heads=heads, selected_steps=sorted(selected_steps),
                      samples=sum(key[0] == index for key in written), representation="post_rope_query")
        (args.output / f"collection_prompt{index}.json").write_text(json.dumps(config, indent=2) + "\n")
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
