#!/usr/bin/env python3
"""Fold the LoRA adapter into the base model once, offline.

Loading the VLM at runtime costs two things this script removes:

  * `from peft import PeftModel` drags in `peft.auto`, which imports the whole
    AutoModel registry the node never uses. Measured at ~14 s of the ~17 s peft
    import tree, and peft cannot be avoided by importing a submodule --
    `peft/__init__.py` runs `from .auto import ...` whichever way you enter.
  * `PeftModel.from_pretrained()` then attaches the adapter at every launch.

WHERE the merge happens decides whether the result is usable. Merging straight
into the 4-bit checkpoint (dequantize -> add delta -> requantize) was measured
at 0/12 agreement with base+adapter: this adapter is r=16 alpha=32, and the
delta is large enough that the NF4 round trip flips the answer. So the merge is
done against the fp16 base, exactly once, and only the merged result is
quantized -- the standard QLoRA path.

    python3 social_perception/scripts/merge_vlm_adapter.py \
        --adapter Saved_Model --output Saved_Model_merged

Always confirm with compare_vlm_merge.py before pointing the node at --output.
"""
import argparse
import gc
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='Saved_Model')
    parser.add_argument('--output', default='Saved_Model_merged')
    # The fp16 checkpoint, NOT the bnb-4bit one the node loads at runtime.
    parser.add_argument('--base', default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--keep-fp16', action='store_true',
                        help='Keep the intermediate fp16 merge on disk (~4.4 GB)')
    arguments = parser.parse_args()

    adapter_path = Path(arguments.adapter).resolve()
    output_path = Path(arguments.output).resolve()
    fp16_path = output_path.with_name(output_path.name + '_fp16')
    if not (adapter_path / 'adapter_config.json').is_file():
        raise SystemExit(f'No adapter_config.json under {adapter_path}')

    import torch
    from peft import PeftModel
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)

    started = time.monotonic()

    # Step 1: merge in fp16 on the CPU. 2B fp16 is ~4.4 GB, which does not fit
    # beside anything else on a 4 GiB card, and the merge is a one-off.
    print(f'Loading fp16 base {arguments.base} on CPU ...', flush=True)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        arguments.base, torch_dtype=torch.float16, device_map=None,
        local_files_only=True)
    print(f'  loaded in {time.monotonic() - started:.1f}s', flush=True)

    print('Attaching adapter ...', flush=True)
    model = PeftModel.from_pretrained(base, str(adapter_path),
                                      torch_dtype=torch.float16)

    print('Merging LoRA into fp16 weights ...', flush=True)
    model = model.merge_and_unload()
    model.eval()

    if fp16_path.exists():
        shutil.rmtree(fp16_path)
    print(f'Saving fp16 merge to {fp16_path} ...', flush=True)
    model.save_pretrained(str(fp16_path), safe_serialization=True)

    del model, base
    gc.collect()

    # Step 2: quantize the merged weights once, so the node loads a checkpoint
    # that is already 4-bit instead of quantizing at every launch.
    print('Quantizing merged model to 4-bit ...', flush=True)
    quantized = Qwen2VLForConditionalGeneration.from_pretrained(
        str(fp16_path), device_map={'': 0}, torch_dtype=torch.float16,
        local_files_only=True,
        # Match the quantisation unsloth shipped the base model with, field for
        # field. Two of these are not cosmetic:
        #
        #  * compute_dtype MUST stay bfloat16. float16 measured 7.76 s of
        #    prefill against 2.31 s for bfloat16 on this Turing card -- same
        #    answers, 3.3x the time, because bitsandbytes picks a slower path.
        #    Note the node's own float16 request is ignored for an already
        #    quantised checkpoint: what is written here is what runs.
        #  * skip_modules keeps lm_head (151k vocab) and the vision merger out
        #    of 4-bit, as unsloth does.
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=['lm_head', 'multi_modal_projector',
                                   'merger', 'modality_projection'],
        ))
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    print(f'Saving 4-bit model to {output_path} ...', flush=True)
    quantized.save_pretrained(str(output_path), safe_serialization=True)

    # The node reads the processor from the model directory, so it has to ship
    # with the merge or every launch falls back to a second lookup.
    try:
        processor = AutoProcessor.from_pretrained(
            str(adapter_path), local_files_only=True)
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(
            arguments.base, local_files_only=True)
    processor.save_pretrained(str(output_path))

    if not arguments.keep_fp16:
        shutil.rmtree(fp16_path)

    total = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    print(f'Done in {time.monotonic() - started:.1f}s, '
          f'{total / 1e9:.2f} GB at {output_path}')


main()
