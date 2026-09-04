#!/usr/bin/env python3
"""Check that the merged VLM answers identically to base+adapter.

Merging LoRA into 4-bit weights is a dequantize/requantize round trip, so the
merged model is not bit-identical. This runs both models over the same crops
with the node's prompt, prefix and decoding settings, and reports every
disagreement plus how confident each answer was.

    python3 social_perception/scripts/compare_vlm_merge.py \
        --crops /path/to/crops --merged Saved_Model_merged
"""
import argparse
import glob
import time
from pathlib import Path

ANSWER_PREFIX = '{"talking":"'      # social_vlm_perception.py:214
PREFIX_ANSWER_TOKENS = 3            # social_vlm_perception.py:217
MIN_PIXELS = 3136                   # vlm_min_pixels
MAX_PIXELS = 200704                 # vlm_max_pixels
PROMPT = ('Quan sát hai người trong ảnh. Họ có đang nói chuyện trực tiếp với '
          'nhau không? Chỉ trả lời đúng một JSON: {"talking":"có"} hoặc '
          '{"talking":"không"}.')


def build_inputs(processor, image, torch):
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': PROMPT}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
    inputs = processor(text=[text], images=[image], padding=False,
                       return_tensors='pt')
    return {key: value.to('cuda:0') for key, value in inputs.items()}


def answer_for(model, processor, image, torch):
    inputs = build_inputs(processor, image, torch)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=PREFIX_ANSWER_TOKENS,
                                do_sample=False, output_scores=True,
                                return_dict_in_generate=True)
    elapsed = time.monotonic() - started
    generated = output.sequences[:, inputs['input_ids'].shape[1]:]
    word = processor.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip().split('"')[0].strip()
    # Confidence of the decisive first token, so a "same answer, but barely"
    # case is visible rather than hidden behind matching strings.
    probabilities = torch.softmax(output.scores[0][0].float(), dim=-1)
    return word, float(probabilities.max()), elapsed


def run(load_model, crops, torch):
    from transformers import AutoProcessor
    model, processor_source = load_model()
    processor = AutoProcessor.from_pretrained(
        processor_source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        local_files_only=True)
    model.eval()
    results = []
    from PIL import Image as PilImage
    for path in crops:
        image = PilImage.open(path).convert('RGB')
        results.append((Path(path).name,) + answer_for(model, processor, image, torch))
    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops', required=True, nargs='+')
    parser.add_argument('--adapter', default='Saved_Model')
    parser.add_argument('--merged', default='Saved_Model_merged')
    parser.add_argument('--base', default='unsloth/Qwen2-VL-2B-Instruct-bnb-4bit')
    parser.add_argument('--merged-first', action='store_true',
                        help='Run the merged model first, to expose order effects')
    arguments = parser.parse_args()

    crops = sorted(f for pattern in arguments.crops
                   for f in glob.glob(f'{pattern}/*.png'))
    if not crops:
        raise SystemExit('No crops found')
    print(f'{len(crops)} crops\n')

    import torch
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    quantization = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.float16)

    def load_reference():
        from peft import PeftModel
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            arguments.base, device_map={'': 0}, torch_dtype=torch.float16,
            local_files_only=True, quantization_config=quantization)
        return PeftModel.from_pretrained(base, arguments.adapter), arguments.base

    def load_merged():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            arguments.merged, device_map={'': 0}, torch_dtype=torch.float16,
            local_files_only=True)
        return model, arguments.merged

    # Whichever model runs second inherits a warmed-up GPU, so the timings are
    # only comparable if the order can be flipped and the result reproduced.
    if arguments.merged_first:
        print('--- merged ---', flush=True)
        merged = run(load_merged, crops, torch)
        print('--- base + adapter (reference) ---', flush=True)
        reference = run(load_reference, crops, torch)
    else:
        print('--- base + adapter (reference) ---', flush=True)
        reference = run(load_reference, crops, torch)
        print('--- merged ---', flush=True)
        merged = run(load_merged, crops, torch)

    disagreements = 0
    print(f'\n{"crop":<20}{"reference":<18}{"merged":<18}{"match"}')
    for (name, ref_word, ref_p, _), (_, new_word, new_p, _) in zip(reference, merged):
        same = ref_word == new_word
        disagreements += not same
        print(f'{name:<20}{ref_word + f" ({ref_p:.2f})":<18}'
              f'{new_word + f" ({new_p:.2f})":<18}{"OK" if same else "DIFFER"}')

    ref_time = sum(r[3] for r in reference) / len(reference)
    new_time = sum(r[3] for r in merged) / len(merged)
    print(f'\nagreement : {len(crops) - disagreements}/{len(crops)}')
    print(f'mean infer: reference {ref_time:.2f}s | merged {new_time:.2f}s')


main()
