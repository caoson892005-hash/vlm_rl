#!/usr/bin/env python3
"""Qwen2-VL loading and inference, kept apart from the perception node.

Split out so the two halves of the pipeline can be installed separately. The
robot runs social_vlm_perception.py, which needs ultralytics but never imports
this module when `vlm_remote` is set; the workstation runs social_vlm_worker.py,
which imports this and needs transformers/peft but no YOLO. Every heavy import
still happens inside VlmBackend.__init__, so importing this module alone costs
nothing.
"""

import contextlib
import os
import threading
import time
from enum import IntEnum

import cv2
import numpy as np


class VlmProgress:
    """Log stage-based VLM progress while blocking native operations run.

    Transformers does not expose byte-level progress callbacks for cached
    checkpoints. Percentages therefore describe completed initialization
    stages. A heartbeat repeats the current stage and elapsed time so a long
    ``from_pretrained`` call is visibly alive without pretending to know its
    remaining duration.
    """

    def __init__(self, logger, label, initial_stage,
                 heartbeat_seconds=10.0, bar_width=20):
        self.logger = logger
        self.label = label
        self.heartbeat_seconds = heartbeat_seconds
        self.bar_width = bar_width
        self.started_at = time.monotonic()
        self.percent = 0
        self.stage = initial_stage
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self._log()
        self.thread = threading.Thread(
            target=self._heartbeat,
            name='vlm-load-progress',
            daemon=True,
        )
        self.thread.start()

    def update(self, percent, stage):
        with self.lock:
            self.percent = max(self.percent, min(100, int(percent)))
            self.stage = stage
        self._log()

    def complete(self, stage):
        self.update(100, stage)
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def fail(self, error):
        with self.lock:
            self.stage = f'FAILED: {type(error).__name__}: {error}'
        self._log()
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _heartbeat(self):
        while not self.stop_event.wait(self.heartbeat_seconds):
            self._log()

    def _log(self):
        with self.lock:
            percent = self.percent
            stage = self.stage
        completed = int(round(self.bar_width * percent / 100.0))
        bar = '#' * completed + '-' * (self.bar_width - completed)
        elapsed = time.monotonic() - self.started_at
        try:
            self.logger.info(
                f'{self.label} [{bar}] {percent:3d}% | {stage} | '
                f'elapsed={elapsed:.0f}s')
        except Exception:
            # Shutdown may invalidate the ROS context while a native model
            # loader is still returning from a background thread.
            pass


class VlmBackend:
    """Standard Transformers/PEFT loader for the supplied Qwen2-VL adapter."""

    # Generation, not the image, dominates latency on a small GPU: measured on
    # a Quadro T1000 the prefill costs ~1.8 s and every further decode step
    # ~0.36 s, so the 7-token answer `{"talking":"Không"}` spent ~2.2 s writing
    # punctuation the prompt already dictates. Teacher-forcing that punctuation
    # leaves only the decisive word to generate.
    ANSWER_PREFIX = '{"talking":"'
    # Enough for the longest tokenization of either word ('KH','Ô','NG'); the
    # shorter ones simply run into the closing quote, which is cut off below.
    PREFIX_ANSWER_TOKENS = 3

    def __init__(self, adapter_path, base_model, load_in_4bit, require_cuda,
                 max_new_tokens, min_pixels, max_pixels, logger,
                 execution_lock=None, force_answer_prefix=True, offline=True,
                 merged_path=None):
        self.logger = logger
        self.execution_lock = execution_lock
        self.force_answer_prefix = force_answer_prefix
        progress = VlmProgress(
            logger, 'VLM LOAD', 'Starting VLM initialization')
        progress.start()
        try:
            if offline:
                # The 2.1 GiB checkpoint is already in ~/.cache/huggingface, but
                # every from_pretrained still asks huggingface.co whether the
                # cached etag is current before using it. That round trip buys
                # nothing on a robot running a pinned model and blocks startup
                # for as long as the network takes to answer -- on a captive
                # Wi-Fi portal or an offline robot, until the HTTP timeout.
                # The env vars cover libraries that resolve files themselves;
                # local_files_only below is what makes the guarantee, since it
                # does not depend on huggingface_hub being imported after this.
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
            progress.update(5, 'Checking Pillow compatibility')
            # Ubuntu 22.04 provides Pillow 9.0.1, whose resampling constants
            # live directly on PIL.Image. New Transformers expects the enum
            # introduced in Pillow 9.1. This alias is API-compatible and
            # avoids a misleading downstream PEFT import error.
            from PIL import Image as PilImageModule
            if not hasattr(PilImageModule, 'Resampling'):
                class PillowResampling(IntEnum):
                    NEAREST = PilImageModule.NEAREST
                    LANCZOS = PilImageModule.LANCZOS
                    BILINEAR = PilImageModule.BILINEAR
                    BICUBIC = PilImageModule.BICUBIC
                    BOX = PilImageModule.BOX
                    HAMMING = PilImageModule.HAMMING

                PilImageModule.Resampling = PillowResampling

            progress.update(10, 'Importing PyTorch')
            import torch
            # A merged checkpoint needs no PEFT at all, and skipping that import
            # is most of the win: `peft` costs ~17 s of the import tree on this
            # machine and cannot be trimmed, because `peft/__init__.py` pulls in
            # the whole AutoModel registry via `from .auto import ...` no matter
            # which submodule is requested. Built by merge_vlm_adapter.py.
            use_merged = merged_path is not None
            progress.update(
                20, 'Importing Transformers' if use_merged
                else 'Importing PEFT and Transformers')
            if not use_merged:
                from peft import PeftModel
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from transformers.utils import logging as transformers_logging
            transformers_logging.set_verbosity_error()
            progress.update(30, 'ML libraries imported')

            self.torch = torch
            self.max_new_tokens = max_new_tokens
            self.min_pixels = min_pixels
            self.max_pixels = max_pixels
            use_cuda = torch.cuda.is_available()
            progress.update(
                32, f'Runtime ready: CUDA={use_cuda}, CUDA runtime={torch.version.cuda}')
            if require_cuda and not use_cuda:
                raise RuntimeError(
                    'CUDA is required for VLM inference but is unavailable')
            model_kwargs = {
                'device_map': 'auto' if use_cuda else None,
                'torch_dtype': torch.float16 if use_cuda else torch.float32,
                'local_files_only': offline,
            }
            if load_in_4bit and use_cuda:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif load_in_4bit:
                logger.warn(
                    'CUDA is unavailable; VLM will attempt a local CPU fallback, '
                    'which can be very slow for Qwen2-VL')

            if use_merged:
                # The merged checkpoint carries its own quantization_config,
                # and for an already-quantised model that config wins over
                # anything passed here -- compute_dtype included. Dropping ours
                # keeps the code honest about who decides: merge_vlm_adapter.py.
                model_kwargs.pop('quantization_config', None)
                if use_cuda:
                    # 'auto' sizes the split against free VRAM at this moment,
                    # and by now YOLO holds some of the card (plus ~450 MiB if
                    # gzserver is PRIME-offloaded onto it). That made accelerate
                    # try to put layers on the CPU, which bitsandbytes rejects
                    # outright. The merged checkpoint is 1.56 GiB and is meant
                    # to sit entirely on the card, so say so: an honest OOM here
                    # beats a partial dispatch that cannot run anyway.
                    model_kwargs['device_map'] = {'': 0}
                progress.update(35, f'Loading merged Qwen2-VL: {merged_path}')
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(merged_path), **model_kwargs)
                self.model.eval()
                self._warn_if_offloaded(self.model, logger)
                progress.update(88, 'Merged model loaded; loading processor')
                processor_sources = (str(merged_path), base_model)
            else:
                progress.update(
                    35,
                    f'Loading Qwen2-VL base model: {base_model}'
                    f'{" (cache only)" if offline else ""}')
                try:
                    base = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model, **model_kwargs)
                except OSError as error:
                    # Only reachable the first time a machine runs this model,
                    # or after the cache is cleared. One download is worth more
                    # than a dead VLM, so pay for it once and stay on the cache.
                    if not offline:
                        raise
                    logger.warn(
                        f'{base_model} is not in the local Hugging Face cache '
                        f'({type(error).__name__}); downloading it once. '
                        'Subsequent launches will load from the cache offline.')
                    progress.update(35, f'Downloading base model: {base_model}')
                    self._disable_offline(model_kwargs)
                    base = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model, **model_kwargs)
                    offline = False
                self._warn_if_offloaded(base, logger)
                progress.update(75, 'Base model loaded; attaching LoRA adapter')
                self.model = PeftModel.from_pretrained(base, str(adapter_path))
                self.model.eval()
                progress.update(88, 'LoRA adapter attached; loading processor')
                processor_sources = (str(adapter_path), base_model)

            processor_kwargs = {
                'min_pixels': self.min_pixels,
                'max_pixels': self.max_pixels,
                'local_files_only': offline,
            }
            primary_source, fallback_source = processor_sources
            try:
                self.processor = AutoProcessor.from_pretrained(
                    primary_source, **processor_kwargs)
            except (OSError, ValueError):
                # The adapter directory ships no preprocessor_config.json, so
                # this fallback is the normal path, not an error case: it is a
                # second trip to the hub unless it is pinned to the cache. The
                # merged directory does ship one, so it does not come here.
                progress.update(92, 'Loading processor from base model')
                self.processor = AutoProcessor.from_pretrained(
                    fallback_source, **processor_kwargs)
            progress.update(97, 'Processor loaded; selecting inference device')
            self.input_device = next(self.model.parameters()).device
            progress.update(98, 'Warming up CUDA kernels')
            self._warm_up()
            progress.complete(
                f'VLM READY: {"merged model" if use_merged else "adapter"} '
                f'loaded on {self.input_device}')
        except Exception as error:
            progress.fail(error)
            raise

    @staticmethod
    def _warn_if_offloaded(model, logger):
        """Say so when the card was too small to hold the whole model."""
        device_map = getattr(model, 'hf_device_map', {})
        offloaded_devices = sorted({
            str(device) for device in device_map.values()
            if str(device) in ('cpu', 'disk')
        })
        if offloaded_devices:
            logger.warn(
                'Part of the VLM was offloaded to '
                f'{", ".join(offloaded_devices)}; inference will be slower')

    @staticmethod
    def _disable_offline(model_kwargs):
        """Reopen the network after a cache miss, for this process only."""
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        model_kwargs['local_files_only'] = False
        try:
            import huggingface_hub.constants as hub_constants
            # Already imported by this point, so it captured the offline flag
            # set above; the env var alone will not take it back.
            hub_constants.HF_HUB_OFFLINE = False
        except Exception:  # A layout change here must not block the download.
            pass

    def _warm_up(self):
        """Pay the one-off CUDA/bitsandbytes kernel cost before the first pair.

        The first generate() call measured 9.4 s against 4.1 s for every call
        after it. Spending that here means the first real conversation is
        judged at the steady-state latency instead of waiting out kernel
        autotuning while two people stand in front of the camera.
        """
        started_at = time.monotonic()
        try:
            blank = np.full((256, 384, 3), 127, dtype=np.uint8)
            self.infer(blank, 'warm-up')
        except Exception as error:  # A failed warm-up must not block startup.
            self.logger.warn(
                f'VLM warm-up failed ({type(error).__name__}: {error}); '
                'the first inference will be slower')
            return
        self.logger.info(
            f'VLM warm-up finished in {time.monotonic() - started_at:.2f}s')

    def infer(self, bgr_image, prompt, pair=None):
        from PIL import Image as PilImage

        started_at = time.monotonic()
        image = PilImage.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        max_new_tokens = self.max_new_tokens
        if self.force_answer_prefix:
            # Start the assistant turn mid-answer so the model resumes from the
            # forced prefix. It only has to produce the word, which also ends
            # the malformed replies (`{"talking":"có}`, `{"talking":true}`)
            # that used to cost a whole inference to recover from.
            text += self.ANSWER_PREFIX
            max_new_tokens = self.PREFIX_ANSWER_TOKENS
        # This is a one-item batch, so padding is unnecessary and only causes
        # a noisy Transformers max_length warning.
        inputs = self.processor(
            text=[text], images=[image], padding=False, return_tensors='pt')
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        lock_context = (self.execution_lock if self.execution_lock is not None
                        else contextlib.nullcontext())
        # Split the wait into preprocessing, waiting for the lock and the model
        # itself. The same crop takes seconds standalone and far longer in the
        # running node, and only a breakdown says which of the three grew.
        prepared_at = time.monotonic()
        with lock_context:
            acquired_at = time.monotonic()
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        self.last_timing = (prepared_at - started_at,
                            acquired_at - prepared_at,
                            time.monotonic() - acquired_at,
                            int(inputs['input_ids'].shape[1]))
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()
        if self.force_answer_prefix:
            # Reattach what was teacher-forced so the published response stays
            # the same JSON object every consumer already parses and logs. The
            # word is the model's; only the syntax around it was supplied.
            word = answer.split('"')[0].strip()
            answer = self.ANSWER_PREFIX + word + '"}'
        return answer
