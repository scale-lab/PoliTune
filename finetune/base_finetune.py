# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in LICENSE-BSD.
#
# Modifications Copyright (c) SCALE Lab, Brown University.
# These modifications are licensed under the MIT license (see LICENSE).

import csv
from tqdm import tqdm
from finetune.utils import pc_instruction, pc_questions_txt_file, custom_prompts, format_instruction, eval_pc, eval_custom_prompts
from torchtune.modules import KVCache
from torchtune.data import AlpacaInstructTemplate
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.datasets import ConcatDataset
from torchtune import config, modules, utils
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Optimizer
from torch import nn
from omegaconf import DictConfig, ListConfig
import torch
from warnings import warn
from typing import Any, Dict, Optional, Tuple
from functools import partial
import sys
import time
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, dir_path)
sys.path.insert(0, parent_dir_path)


log = utils.get_logger("DEBUG")


class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.g

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.

    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if (
            self._dtype == torch.bfloat16
            and self._device != torch.device("cpu")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "Full bf16 training is not supported on this hardware.")
        # logging attributes
        self._output_dir = cfg.output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self._template = AlpacaInstructTemplate()
        self._pc_instruction = pc_instruction
        self._pc_questions = []
        with open(pc_questions_txt_file, "r") as f:
            for line in f:
                self._pc_questions.append(
                    self.generate_pc_instruction(line.strip()))

        self._custom_prompts = custom_prompts
        self._custom_prompts = [self.format_instruction(
            q) for q in self._custom_prompts]

        self._pc_num_questions = len(self._pc_questions)
        self._pc_csv_file = f"{self._output_dir}/pc.csv"
        self._pc_headers = ['iteration', 'step'] + \
            [f"question_{str(i)}" for i in range(self._pc_num_questions)]
        with open(self._pc_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._pc_headers)

        self._custom_prompts_file = f"{self._output_dir}/custom_instrs.csv"
        headers = ['iteration', 'step'] + \
            [f"prompt_{str(i)}" for i in range(len(self._custom_prompts))]
        with open(self._custom_prompts_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        self._eval_freq = cfg.get("eval_freq", 512)
        self._max_generated_tokens = 300
        self._temperature = 0.3
        self._top_k = 200

        log.info(f"Evaluation frequency: {self._eval_freq}")
        log.info(f"Max generated tokens: {self._max_generated_tokens}")
        log.info(f"Temperature: {self._temperature}")
        log.info(f"Top k: {self._top_k}")

    def format_instruction(self, instr, inp=""):
        return format_instruction(self._template, instr, inp)

    def generate_pc_instruction(self, question):
        return self.format_instruction(self._pc_instruction, question)

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            if utils.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[utils.SEED_KEY]
            or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            or self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]
        ):
            warn(
                message="""Configured value for seed, epochs or max_steps_per_epoch
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]
        self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._model_compile = cfg.compile
        checkpoint_dict = self.load_checkpoint(
            cfg_checkpointer=cfg.checkpointer)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[utils.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[utils.OPT_KEY] if self._resume_from_checkpoint else None
            ),
        )

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        self._profiler_enabled = cfg.profiler.enabled
        self._profiler = config.instantiate(cfg.profiler)
        self._pc_questions = [torch.tensor(self._tokenizer.encode(
            q, add_bos=True, add_eos=False), dtype=torch.int, device=self._device).to(self._device) for q in self._pc_questions[:]]
        self._custom_prompts = [torch.tensor(self._tokenizer.encode(
            q, add_bos=True, add_eos=False), dtype=torch.int, device=self._device).to(self._device) for q in self._custom_prompts[:]]
        self._causal_mask = torch.tril(
            torch.ones(self._model.max_seq_len,
                       self._model.max_seq_len, dtype=torch.bool)
        ).to(self._device)
        self._kv_cache = []
        for _ in self._model.layers:
            self._kv_cache.append(KVCache(
                batch_size=1,
                max_seq_len=self._model.max_seq_len,
                num_heads=self._model.num_heads,
                head_dim=self._model.head_dim,
                dtype=self._dtype,
            ).to(self._device))

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=cfg_model.lora_attn_modules,
            apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
            apply_lora_to_output=getattr(
                cfg_model, "apply_lora_to_output", False),
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Validate model adapter params were loaded in with the expected dtype
        # TODO (rohan-varma): Further validation to ensure the appropriate base params
        # are NF4 vs bf16 based on the quantization config.
        utils.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        log.info(f"Model is initialized with precision {self._dtype}.")
        # Compile model, if enabled.
        if compile_model:
            log.info("Compiling model with torch.compile...")
            model = utils.wrap_compile(model)
        if self._device.type == "cuda":
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset,
                                   tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            ),
        )

        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def eval_pc(self, iteration=0, step=0):
        return eval_pc(pc_questions=self._pc_questions, pc_csv_file=self._pc_csv_file, log=log, model=self._model, tokenizer=self._tokenizer, causal_mask=self._causal_mask, kv_cache=self._kv_cache, max_generated_tokens=self._max_generated_tokens, temperature=self._temperature, top_k=self._top_k, iteration=iteration, step=step)

    def eval_custom_prompts(self, iteration=0, step=0):
        return eval_custom_prompts(custom_prompts=self._custom_prompts, custom_prompts_file=self._custom_prompts_file, log=log, model=self._model, tokenizer=self._tokenizer, causal_mask=self._causal_mask, kv_cache=self._kv_cache, max_generated_tokens=self._max_generated_tokens, temperature=self._temperature, top_k=self._top_k, iteration=iteration, step=step)

    def save_checkpoint(self, step: str) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}
        # if training is in-progress, checkpoint the optimizer state as well
        # if epoch + 1 < self.total_epochs:
        ckpt_dict.update(
            {
                # utils.OPT_KEY: self._optimizer.state_dict(),
                utils.SEED_KEY: self.seed,
                utils.EPOCHS_KEY: self.epochs_run,
                utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
            }
        )

        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

        # Construct the full state dict with LoRA weights merged into base LLM weights
        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt_dict.update({utils.MODEL_KEY: {}})

        # Construct the adapter weights
        def adapter_key_filter(x): return x in self.adapter_params
        adapter_state_dict = {
            k: v for k, v in self._model.state_dict().items() if adapter_key_filter(k)
        }
        ckpt_dict.update({utils.ADAPTER_KEY: adapter_state_dict})
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=step,
            intermediate_checkpoint=False,
        )

    def train(self) -> None:
        """
        The core training loop.
        """

        if self._model_compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        log.info(
            f"Starting training from epoch {self.epochs_run + 1}/{self.total_epochs}")
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            # Optionally profile the training loop
            with self._profiler:
                pbar = tqdm(total=self._steps_per_epoch)
                log.info(f"Number of samples: {len(self._dataloader)}")

                for idx, batch in enumerate(self._dataloader):
                    if (
                        self.max_steps_per_epoch is not None
                        and (idx // self._gradient_accumulation_steps)
                        == self.max_steps_per_epoch
                    ):
                        break

                    if self._profiler_enabled:
                        self._profiler.step()

                    input_ids, labels = batch
                    input_ids = input_ids.to(self._device)
                    num_tokens += input_ids.numel()
                    labels = labels.to(self._device)
                    logits = self._model(input_ids)
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    logits = logits.transpose(1, 2)
                    # Compute loss
                    loss = self._loss_fn(logits, labels)
                    loss = loss / self._gradient_accumulation_steps
                    running_loss += loss
                    loss.backward()

                    # Step with optimizer
                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()
                        # Update the number of steps when the weights are updated
                        self.global_step += 1

                        loss_to_log = running_loss.item()
                        pbar.update(1)
                        pbar.set_description(
                            f"{curr_epoch+1}|{self.global_step}|Loss: {loss_to_log}"
                        )

                        # Reset running stats for the next step
                        running_loss = 0
                        num_tokens = 0
                        t0 = time.perf_counter()

            self.epochs_run += 1

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRAFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
