#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("PiscesLx.Tools.Train.Preference")

class PiscesLxToolsPreferenceTrainer:
    """Unifies SFT/DPO/PPO preference alignment under a single facade.

    To preserve existing behavior, this class delegates to the legacy
    tools/rlhf.py::rlhf_train(args) until fully migrated.
    """

    def __init__(self, cfg, hooks, profiler, args=None):
        """Initialize the preference trainer.

        Args:
            cfg: Configuration object.
            hooks: Hooks to be used during training.
            profiler: Profiler for performance analysis.
            args (optional): Legacy arguments. Defaults to None.
        """
        self.cfg = cfg
        self.hooks = hooks
        self.profiler = profiler
        self.args = args

    def run_sft(self, cfg) -> None:
        """Run the SFT (Supervised Fine-Tuning) mode with watermark integration."""
        self._run_sft(self.args or cfg)

    def run_dpo(self, cfg) -> None:
        """Run the DPO (Direct Preference Optimization) mode with watermark integration."""
        self._run_dpo(self.args or cfg)

    def run_ppo(self, cfg) -> None:
        """Run the PPO (Proximal Policy Optimization) mode.

        Args:
            cfg: Configuration object.
        """
        self._delegate_legacy("ppo")

    def _delegate_legacy(self, mode: str) -> None:
        """Integrated RLHF (PPO) implementation (legacy-free).

        Args:
            mode (str): 'sft' | 'dpo' | 'ppo' (currently we run PPO as default)
        """
        # For now, we provide PPO path; SFT/DPO hooks can be added similarly if needed
        try:
            self._run_rlhf_ppo(self.args)
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"RLHF failed: {e}")
            raise

    # ---------------- RLHF (PPO) integrated implementation ----------------
    def _run_rlhf_ppo(self, args) -> None:
        """Run PPO-based RLHF training, migrated from tools/rlhf.py (behavior-preserving)."""
        import os
        import re
        import gc
        import torch
        import torch.nn as nn
        from trl import PPOTrainer, PPOConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.success("Starting RLHF training with PPO (integrated)...")

        # -------- Validate and normalize args --------
        args = self._validate_rlhf_args(args)

        if not hasattr(args, 'model_path') or not args.model_path:
            logger.error("Model path --model_path must be specified")
            raise SystemExit(1)

        # -------- Load model/tokenizer --------
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # -------- PPO config --------
        ppo_config = PPOConfig(
            model_name=args.model_path,
            learning_rate=args.rlhf_lr,
            batch_size=args.rlhf_batch_size,
            mini_batch_size=args.rlhf_mini_batch_size,
            gradient_accumulation_steps=args.rlhf_accum_steps,
            optimize_cuda_cache=True,
            log_with="wandb" if getattr(args, 'log_with_wandb', False) else None,
            max_grad_norm=1.0,
            target_kl=0.1,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            cliprange_value=0.2,
            cliprange=0.2,
            early_stopping=True,
        )

        ppo_trainer = PPOTrainer(model=model, config=ppo_config, tokenizer=tokenizer, dataset=None)

        # -------- Load dataset --------
        try:
            from datasets import load_dataset
            dataset = load_dataset(args.rlhf_dataset, split="train")
            logger.success(f"Loaded RLHF dataset: {args.rlhf_dataset}")
        except Exception as e:
            logger.error(f"Failed to load RLHF dataset: {e}")
            raise SystemExit(1)

        # -------- Reward function --------
        def compute_reward(completions, human_feedback, ai_feedback=None, safety_score=None, adversarial_score=None):
            import torch
            # Human
            human_rewards = torch.tensor([float(score) for score in human_feedback], dtype=torch.float32)
            human_rewards = torch.nan_to_num(human_rewards, nan=1.0, posinf=2.0, neginf=0.0)
            # AI feedback (rule-based if not provided)
            if ai_feedback is None:
                ai_feedback = []
                for completion in completions:
                    try:
                        response_text = tokenizer.decode(completion, skip_special_tokens=True)
                        quality_score = 1.0
                        word_count = len(response_text.split())
                        if 10 <= word_count <= 200:
                            quality_score += 0.2 * min(word_count / 50.0, 1.0)
                        reasoning_keywords = ["explain", "because", "therefore", "consequently", "as a result"]
                        reasoning_count = sum(1 for kw in reasoning_keywords if kw in response_text.lower())
                        quality_score += 0.3 * min(reasoning_count / 3.0, 1.0)
                        sentence_count = response_text.count(".")
                        if sentence_count > 2:
                            quality_score += 0.1 * min(sentence_count / 5.0, 1.0)
                        unique_words = len(set(response_text.lower().split()))
                        total_words = max(len(response_text.split()), 1)
                        diversity_ratio = unique_words / total_words
                        quality_score += 0.2 * diversity_ratio
                        ai_feedback.append(min(quality_score, 2.0))
                    except Exception:
                        ai_feedback.append(1.0)
            ai_rewards = torch.tensor([float(score) for score in ai_feedback], dtype=torch.float32)
            ai_rewards = torch.nan_to_num(ai_rewards, nan=1.0, posinf=2.0, neginf=0.0)
            # Safety
            if safety_score is None:
                safety_rewards = torch.ones_like(human_rewards)
                harmful_categories = {
                    "physical_harm": ["harm", "danger", "injury", "violence"],
                    "illegal_activities": ["illegal", "crime", "steal", "fraud"],
                    "toxic_content": ["hate", "discrimination", "racism", "sexism"],
                    "misinformation": ["false", "misleading", "fake", "conspiracy"],
                }
                for i, completion in enumerate(completions):
                    try:
                        response_text = tokenizer.decode(completion, skip_special_tokens=True)
                        safety_penalty = 0.0
                        for _, keywords in harmful_categories.items():
                            if any(kw in response_text.lower() for kw in keywords):
                                safety_penalty += 0.3
                        safety_rewards[i] = max(0.1, 1.0 - safety_penalty)
                    except Exception:
                        safety_rewards[i] = 0.5
            else:
                safety_rewards = torch.tensor([float(score) for score in safety_score], dtype=torch.float32)
                safety_rewards = torch.nan_to_num(safety_rewards, nan=1.0, posinf=1.0, neginf=0.1)
            # Adversarial
            if adversarial_score is None:
                adversarial_rewards = torch.ones_like(human_rewards)
                patterns = [r"ignore.*previous.*instructions", r"disregard.*safety.*guidelines", r"bypass.*security.*measures", r"override.*constraints"]
                for i, completion in enumerate(completions):
                    try:
                        response_text = tokenizer.decode(completion, skip_special_tokens=True)
                        robustness = 1.0
                        for pat in patterns:
                            if re.search(pat, response_text, re.IGNORECASE):
                                robustness -= 0.4
                        adversarial_rewards[i] = max(0.1, robustness)
                    except Exception:
                        adversarial_rewards[i] = 0.5
            else:
                adversarial_rewards = torch.tensor([float(score) for score in adversarial_score], dtype=torch.float32)
                adversarial_rewards = torch.nan_to_num(adversarial_rewards, nan=1.0, posinf=1.0, neginf=0.1)
            # Combine
            human_weight, ai_weight, safety_weight, adversarial_weight = 0.6, 0.15, 0.15, 0.1
            temperature = 0.8
            final_rewards = (
                human_weight * human_rewards +
                ai_weight * ai_rewards +
                safety_weight * safety_rewards +
                adversarial_weight * adversarial_rewards
            ) * temperature
            final_rewards = torch.nan_to_num(final_rewards, nan=1.0, posinf=2.0, neginf=0.0)
            return torch.clamp(final_rewards, 0.0, 2.0)

        # -------- PPO training loop (simplified behavior-preserving) --------
        # AMP/mixed precision from utils.device (SSOT)
        try:
            from utils import PiscesLxCoreDeviceFacade
            _dev_cfg_amp = PiscesLxCoreDeviceFacade().setup_devices(mode="train")
            _mp_enabled = bool(_dev_cfg_amp.get('mixed_precision', False))
            _dtype_str = str(_dev_cfg_amp.get('dtype', 'fp16')).lower()
            import torch as _torch
            _amp_dtype = _torch.float16 if _dtype_str == 'fp16' else (_torch.bfloat16 if _dtype_str == 'bf16' else _torch.float32)
        except Exception:
            import torch as _torch
            _mp_enabled = _torch.cuda.is_available()
            _amp_dtype = _torch.float16
        total_steps = 0
        try:
            from datasets import load_dataset
            # Iterate over epochs
            for epoch in range(args.rlhf_epochs):
                epoch_rewards = []
                # Iterate limited samples
                max_len = min(len(dataset), args.rlhf_max_samples)
                for batch_idx, batch in enumerate(dataset.select(range(max_len))):
                    inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True)
                    query_tensors = inputs["input_ids"]
                    # Watermark-aware generation: apply logits processor (lexical control)
                    from utils import PiscesLxUtilsLogitsProcessor as PiscesWatermarkLogitsProcessor
                    wm_seed = int(torch.randint(0, 2**31-1, (1,)).item())
                    logits_proc = [PiscesWatermarkLogitsProcessor(seed=wm_seed, boost=float(self.cfg.get("watermark.boost", 0.15)))]
                    response_tensors = ppo_trainer.generate(
                        query_tensors,
                        max_new_tokens=args.rlhf_max_length,
                        do_sample=True,
                        temperature=0.7,
                        logits_processor=logits_proc
                    )
                    # Add signed payload watermark to decoded text for traceability and monitor detect score
                    try:
                        from utils import PiscesLxUtilsWatermarkManager
                        texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                        _wmgr = PiscesLxUtilsWatermarkManager()
                        texts_wm = [_wmgr.add_watermark(t, {"user_id": "training"}) for t in texts]
                        # Detect and summarize watermark scores for observability
                        try:
                            scores = []
                            for tw in texts_wm:
                                det = _wmgr.check_watermark(tw)
                                s = (det or {}).get("score", None)
                                if s is not None:
                                    scores.append(float(s))
                            if scores:
                                avg_s = sum(scores) / len(scores)
                                logger.debug(f"Training watermark detect score avg: {avg_s:.4f} (n={len(scores)})")
                                try:
                                    self.hooks.emit("train.watermark.score", avg=float(avg_s), count=len(scores))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                    human_feedback = batch.get("human_score", [1.0] * len(response_tensors))
                    ai_feedback = batch.get("ai_score", None)
                    safety_score = batch.get("safety_score", None)
                    adversarial_score = batch.get("adversarial_score", None)
                    rewards = compute_reward(response_tensors, human_feedback, ai_feedback, safety_score, adversarial_score)
                    try:
                        with torch.amp.autocast('cuda', enabled=_mp_enabled, dtype=_amp_dtype):
                            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                            logger.error(f"Numerical instability detected in rewards at batch {batch_idx}")
                            rewards = torch.nan_to_num(rewards, nan=1.0, posinf=2.0, neginf=0.0)
                        epoch_rewards.append(rewards.mean().item())
                        total_steps += 1
                        if total_steps % 10 == 0:
                            logger.debug(f"Step {total_steps}, Reward: {rewards.mean():.4f}, KL: {stats.get('objective/kl', 0):.4f}")
                            if stats.get('objective/kl', 0) > ppo_config.target_kl:
                                logger.warning(f"KL divergence {stats['objective/kl']:.4f} exceeded target {ppo_config.target_kl}, considering early stopping")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.error(f"OOM at batch {batch_idx}, clearing cache and continuing")
                            try:
                                import torch as _torch
                                if _torch.cuda.is_available():
                                    _torch.cuda.empty_cache()
                            except Exception:
                                pass
                            continue
                        else:
                            logger.error(f"Runtime error in batch {batch_idx}: {e}")
                            continue
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
                        continue
                avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
                logger.success(f"Epoch {epoch+1}/{args.rlhf_epochs} completed, Avg Reward: {avg_reward:.4f}")
        except Exception as e:
            logger.error(f"Critical error during RLHF training: {e}")
            raise

        # -------- Save trained model (optional) --------
        if getattr(args, 'output_dir', None):
            output_path = os.path.join(args.output_dir, f"rlhf_model_{args.rlhf_epochs}epochs")
            try:
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                logger.success(f"RLHF model saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                raise
        logger.success("RLHF training completed successfully!")

    # ---------------- SFT integrated implementation ----------------
    def _run_sft(self, args) -> None:
        """Run SFT with watermark-aware generation (logits processor + signed payload + detect score)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            assert hasattr(args, 'model_path') and args.model_path
        except Exception:
            logger.error("SFT requires --model_path")
            raise SystemExit(1)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # Simple demonstration: take a sample prompt from args or fallback
        prompt = getattr(args, 'sft_prompt', None) or "Explain the significance of watermarks in AI-generated content."
        inputs = tokenizer(prompt, return_tensors="pt")
        query_tensors = inputs["input_ids"]

        # Watermark-aware logits processor
        try:
            from utils import PiscesLxUtilsLogitsProcessor as PiscesWatermarkLogitsProcessor
            wm_seed = int(torch.randint(0, 2**31-1, (1,)).item())
            logits_proc = [PiscesWatermarkLogitsProcessor(seed=wm_seed)]
        except Exception:
            logits_proc = None

        # Generate with watermark control
        gen_ids = model.generate(
            query_tensors,
            max_new_tokens=int(getattr(args, 'sft_max_length', 256)),
            do_sample=True,
            temperature=float(getattr(args, 'sft_temperature', 0.7)),
            top_p=float(getattr(args, 'sft_top_p', 0.95)),
            logits_processor=logits_proc
        )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Add signed payload and detect score
        try:
            from utils import PiscesLxUtilsWatermarkManager
            _wmgr = PiscesLxUtilsWatermarkManager()
            text_wm = _wmgr.add_watermark(text, {"user_id": "sft"})
            det = _wmgr.check_watermark(text_wm)
            score = (det or {}).get("score", None)
            if score is not None:
                logger.debug(f"SFT watermark detect score: {float(score):.4f}")
                try:
                    self.hooks.emit("train.sft.watermark.score", score=float(score))
                except Exception:
                    pass
        except Exception:
            pass
        logger.success("SFT generation with watermark completed.")

    # ---------------- DPO integrated implementation ----------------
    def _run_dpo(self, args) -> None:
        """Run DPO with watermark-aware generation (logits processor + signed payload + detect score)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            assert hasattr(args, 'model_path') and args.model_path
        except Exception:
            logger.error("DPO requires --model_path")
            raise SystemExit(1)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # Sample prompt; in real DPO you'd use preference pairs; here we demonstrate generation integration
        prompt = getattr(args, 'dpo_prompt', None) or "Provide a concise summary of AI watermark compliance."
        inputs = tokenizer(prompt, return_tensors="pt")
        query_tensors = inputs["input_ids"]

        # Watermark-aware logits processor
        try:
            from utils import PiscesLxUtilsLogitsProcessor as PiscesWatermarkLogitsProcessor
            wm_seed = int(torch.randint(0, 2**31-1, (1,)).item())
            logits_proc = [PiscesWatermarkLogitsProcessor(seed=wm_seed)]
        except Exception:
            logits_proc = None

        # Generate with watermark control
        gen_ids = model.generate(
            query_tensors,
            max_new_tokens=int(getattr(args, 'dpo_max_length', 256)),
            do_sample=True,
            temperature=float(getattr(args, 'dpo_temperature', 0.7)),
            top_p=float(getattr(args, 'dpo_top_p', 0.95)),
            logits_processor=logits_proc
        )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Add signed payload and detect score
        try:
            from utils import PiscesLxUtilsWatermarkManager
            _wmgr = PiscesLxUtilsWatermarkManager()
            text_wm = _wmgr.add_watermark(text, {"user_id": "dpo"})
            det = _wmgr.check_watermark(text_wm)
            score = (det or {}).get("score", None)
            if score is not None:
                logger.debug(f"DPO watermark detect score: {float(score):.4f}")
                try:
                    self.hooks.emit("train.dpo.watermark.score", score=float(score))
                except Exception:
                    pass
        except Exception:
            pass
        logger.success("DPO generation with watermark completed.")

    def _validate_rlhf_args(self, args):
        """Validate/normalize RLHF arguments (behavior-preserving)."""
        # Defaults
        defaults = {
            'rlhf_lr': 1e-5,
            'rlhf_batch_size': 4,
            'rlhf_mini_batch_size': 1,
            'rlhf_accum_steps': 4,
            'rlhf_epochs': 3,
            'rlhf_max_samples': 1000,
            'rlhf_max_length': 512,
            'rlhf_dataset': 'dunimd/human_feedback',
            'log_with_wandb': False,
            'output_dir': None,
        }
        for k, v in defaults.items():
            if not hasattr(args, k):
                setattr(args, k, v)
        # Ranges
        try:
            if float(args.rlhf_lr) <= 0:
                raise ValueError
        except Exception:
            raise SystemExit("rlhf_lr must be > 0")
        for name in ['rlhf_batch_size', 'rlhf_mini_batch_size', 'rlhf_accum_steps', 'rlhf_epochs', 'rlhf_max_samples', 'rlhf_max_length']:
            try:
                val = int(getattr(args, name))
                if val <= 0:
                    raise ValueError
            except Exception:
                raise SystemExit(f"{name} must be a positive integer")
        return args
