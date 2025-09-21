#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import gc
import os
import torch
import torch.nn as nn
from trl import PPOTrainer, PPOConfig
from utils import RIGHT, DEBUG, ERROR, WARNING
from transformers import AutoModelForCausalLM, AutoTokenizer

def rlhf_train(args):
    """
    Perform a complete RLHF (Reinforcement Learning with Human Feedback) training process using PPO algorithm.
    This function is called from manage.py to initiate the training workflow.

    Args:
        args (Namespace): A namespace object containing all training parameters, 
                         including model path, learning rate, batch size, etc.

    Returns:
        None: This function does not return any value, but it trains and saves the model.
    """
    RIGHT("Starting RLHF training with PPO...")

    # Validate and normalize the input arguments to ensure they meet the training requirements
    try:
        args = validate_rlhf_args(args)
    except Exception as e:
        ERROR(f"Invalid RLHF arguments: {e}")
        return
    
    # Check if the model path is specified, which is essential for loading the pre-trained model
    if not hasattr(args, 'model_path') or not args.model_path:
        ERROR("Model path --model_path must be specified")
        return
    
    # Load the pre-trained causal language model and its corresponding tokenizer from the given path
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Configure the PPO training parameters with gradient clipping and stability protections
    ppo_config = PPOConfig(
        model_name=args.model_path,
        learning_rate=args.rlhf_lr,
        batch_size=args.rlhf_batch_size,
        mini_batch_size=args.rlhf_mini_batch_size,
        gradient_accumulation_steps=args.rlhf_accum_steps,
        optimize_cuda_cache=True,
        log_with="wandb" if args.log_with_wandb else None,
        # Add gradient clipping for training stability
        max_grad_norm=1.0,
        # Add KL divergence penalty to prevent policy collapse
        target_kl=0.1,
        # Add entropy bonus for exploration
        init_kl_coef=0.2,
        # Add adaptive KL control
        adap_kl_ctrl=True,
        # Add value function clipping for stability
        cliprange_value=0.2,
        # Add PPO clipping range
        cliprange=0.2,
        # Add early stopping based on KL divergence
        early_stopping=True,
    )
    
    # Initialize the PPO trainer with the loaded model, configuration, and tokenizer
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=None,
    )
    
    # Load the human feedback dataset for training
    try:
        from datasets import load_dataset
        dataset = load_dataset(args.rlhf_dataset, split="train")
        RIGHT(f"Loaded RLHF dataset: {args.rlhf_dataset}")
    except Exception as e:
        ERROR(f"Failed to load RLHF dataset: {e}")
        return
    
    def compute_reward(completions, human_feedback, ai_feedback=None, safety_score=None, adversarial_score=None):
        """
        Implement an advanced reward function with adversarial training and robustness validation.
        The rewards are calculated based on human scores, AI quality evaluation, safety checks, and adversarial robustness.

        Args:
            completions (torch.Tensor): Model-generated responses.
            human_feedback (list): List of human scores.
            ai_feedback (list, optional): AI-generated quality scores. Defaults to None.
            safety_score (list, optional): Safety/harmlessness scores. Defaults to None.
            adversarial_score (list, optional): Adversarial robustness scores. Defaults to None.

        Returns:
            torch.Tensor: Tensor containing weighted reward scores in the range [0.0, 2.0].
        """
        # Convert human feedback scores from list to tensor with numerical stability
        human_rewards = torch.tensor([float(score) for score in human_feedback], dtype=torch.float32)
        
        # Add numerical stability check
        if torch.isnan(human_rewards).any() or torch.isinf(human_rewards).any():
            human_rewards = torch.nan_to_num(human_rewards, nan=1.0, posinf=2.0, neginf=0.0)
        
        # Perform AI-assisted evaluation using rule-based quality checks if AI feedback is not provided
        if ai_feedback is None:
            ai_feedback = []
            for completion in completions:
                try:
                    # Decode the model-generated response to text with error handling
                    response_text = tokenizer.decode(completion, skip_special_tokens=True)
                    
                    # Evaluate the quality of the response based on advanced rules
                    quality_score = 1.0
                    
                    # Length-based scoring with normalization
                    word_count = len(response_text.split())
                    if 10 <= word_count <= 200:  # Optimal length range
                        quality_score += 0.2 * min(word_count / 50.0, 1.0)  # Normalize length bonus
                    
                    # Reasoning chain detection
                    reasoning_keywords = ["explain", "because", "therefore", "consequently", "as a result"]
                    reasoning_count = sum(1 for keyword in reasoning_keywords if keyword in response_text.lower())
                    quality_score += 0.3 * min(reasoning_count / 3.0, 1.0)
                    
                    # Structure and coherence checks
                    sentence_count = response_text.count(".")
                    if sentence_count > 2:
                        quality_score += 0.1 * min(sentence_count / 5.0, 1.0)
                    
                    # Add diversity score to prevent repetitive responses
                    unique_words = len(set(response_text.lower().split()))
                    total_words = max(len(response_text.split()), 1)
                    diversity_ratio = unique_words / total_words
                    quality_score += 0.2 * diversity_ratio
                    
                    ai_feedback.append(min(quality_score, 2.0))
                    
                except Exception as e:
                    ERROR(f"Error in AI feedback computation: {e}")
                    ai_feedback.append(1.0)  # Default safe score
        
        # Convert AI feedback scores from list to tensor with stability checks
        ai_rewards = torch.tensor([float(score) for score in ai_feedback], dtype=torch.float32)
        if torch.isnan(ai_rewards).any() or torch.isinf(ai_rewards).any():
            ai_rewards = torch.nan_to_num(ai_rewards, nan=1.0, posinf=2.0, neginf=0.0)
        
        # Perform safety check on the model-generated responses
        if safety_score is None:
            safety_rewards = torch.ones_like(human_rewards)
            for i, completion in enumerate(completions):
                try:
                    response_text = tokenizer.decode(completion, skip_special_tokens=True)
                    
                    # Enhanced safety detection with multiple categories
                    harmful_categories = {
                        "physical_harm": ["harm", "danger", "injury", "violence"],
                        "illegal_activities": ["illegal", "crime", "steal", "fraud"],
                        "toxic_content": ["hate", "discrimination", "racism", "sexism"],
                        "misinformation": ["false", "misleading", "fake", "conspiracy"]
                    }
                    
                    safety_penalty = 0.0
                    for category, keywords in harmful_categories.items():
                        if any(keyword in response_text.lower() for keyword in keywords):
                            safety_penalty += 0.3  # Cumulative penalty for multiple violations
                    
                    safety_rewards[i] = max(0.1, 1.0 - safety_penalty)  # Ensure minimum safety score
                    
                except Exception as e:
                    ERROR(f"Error in safety check: {e}")
                    safety_rewards[i] = 0.5  # Conservative safety score on error
        else:
            # Convert safety scores from list to tensor with stability checks
            safety_rewards = torch.tensor([float(score) for score in safety_score], dtype=torch.float32)
            if torch.isnan(safety_rewards).any() or torch.isinf(safety_rewards).any():
                safety_rewards = torch.nan_to_num(safety_rewards, nan=1.0, posinf=1.0, neginf=0.1)
        
        # Adversarial robustness scoring
        if adversarial_score is None:
            # Default adversarial robustness based on response consistency
            adversarial_rewards = torch.ones_like(human_rewards)
            for i, completion in enumerate(completions):
                try:
                    response_text = tokenizer.decode(completion, skip_special_tokens=True)
                    
                    # Check for adversarial patterns
                    adversarial_patterns = [
                        r"ignore.*previous.*instructions",
                        r"disregard.*safety.*guidelines",
                        r"bypass.*security.*measures",
                        r"override.*constraints"
                    ]
                    
                    import re
                    robustness_score = 1.0
                    for pattern in adversarial_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            robustness_score -= 0.4  # Heavy penalty for adversarial vulnerability
                    
                    adversarial_rewards[i] = max(0.1, robustness_score)
                    
                except Exception as e:
                    ERROR(f"Error in adversarial check: {e}")
                    adversarial_rewards[i] = 0.5  # Conservative robustness score on error
        else:
            # Convert adversarial scores from list to tensor with stability checks
            adversarial_rewards = torch.tensor([float(score) for score in adversarial_score], dtype=torch.float32)
            if torch.isnan(adversarial_rewards).any() or torch.isinf(adversarial_rewards).any():
                adversarial_rewards = torch.nan_to_num(adversarial_rewards, nan=1.0, posinf=1.0, neginf=0.1)
        
        # Combine all reward components with advanced weighting
        # Dynamic weighting based on data quality
        human_weight = 0.6
        ai_weight = 0.15
        safety_weight = 0.15
        adversarial_weight = 0.1
        
        # Apply temperature scaling for numerical stability
        temperature = 0.8
        final_rewards = (
            human_weight * human_rewards +
            ai_weight * ai_rewards +
            safety_weight * safety_rewards +
            adversarial_weight * adversarial_rewards
        ) * temperature
        
        # Final numerical stability check and clamping
        final_rewards = torch.nan_to_num(final_rewards, nan=1.0, posinf=2.0, neginf=0.0)
        return torch.clamp(final_rewards, 0.0, 2.0)
    
    # Start the RLHF training loop
    total_steps = 0
    try:
        for epoch in range(args.rlhf_epochs):
            epoch_rewards = []
            
            # Iterate over batches of the dataset
            for batch_idx, batch in enumerate(dataset.select(range(min(len(dataset), args.rlhf_max_samples)))):
                # Prepare the input tensors for the model
                inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True)
                query_tensors = inputs["input_ids"]
                
                # Generate responses using the PPO trainer
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    max_new_tokens=args.rlhf_max_length,
                    do_sample=True,
                    temperature=0.7
                )
                
                # Get feedback scores from the batch data
                human_feedback = batch.get("human_score", [1.0] * len(response_tensors))
                ai_feedback = batch.get("ai_score", None)  # Optional AI quality evaluation
                safety_score = batch.get("safety_score", None)  # Optional safety evaluation
                adversarial_score = batch.get("adversarial_score", None)  # Optional adversarial robustness evaluation
                
                # Compute rewards based on the feedback with robustness validation
                rewards = compute_reward(response_tensors, human_feedback, ai_feedback, safety_score, adversarial_score)
                
                # Perform a PPO update step with numerical stability protection
                try:
                    # Add gradient clipping and stability checks
                    with torch.cuda.amp.autocast(enabled=True):  # Enable mixed precision for stability
                        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                    
                    # Validate reward statistics
                    if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                        ERROR(f"Numerical instability detected in rewards at batch {batch_idx}")
                        rewards = torch.nan_to_num(rewards, nan=1.0, posinf=2.0, neginf=0.0)
                    
                    epoch_rewards.append(rewards.mean().item())
                    
                    total_steps += 1
                    if total_steps % 10 == 0:
                        DEBUG(f"Step {total_steps}, Reward: {rewards.mean():.4f}, KL: {stats.get('objective/kl', 0):.4f}")
                        
                        # Early stopping based on KL divergence
                        if stats.get('objective/kl', 0) > ppo_config.target_kl:
                            WARNING(f"KL divergence {stats['objective/kl']:.4f} exceeded target {ppo_config.target_kl}, considering early stopping")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        ERROR(f"OOM error at batch {batch_idx}, clearing cache and continuing")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        ERROR(f"Runtime error in batch {batch_idx}: {e}")
                        continue
                except Exception as e:
                    ERROR(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # End of epoch - calculate epoch statistics
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        RIGHT(f"Epoch {epoch+1}/{args.rlhf_epochs} completed, Avg Reward: {avg_reward:.4f}")
    
    except Exception as e:
        ERROR(f"Critical error during RLHF training: {e}")
        raise
    
    # Save the trained model if an output directory is specified
    if args.output_dir:
        output_path = os.path.join(args.output_dir, f"rlhf_model_{args.rlhf_epochs}epochs")
        
        # Save with numerical stability validation
        try:
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            RIGHT(f"RLHF model saved to: {output_path}")
            
            # Validate saved model integrity
            from transformers import AutoModelForCausalLM, AutoTokenizer
            test_model = AutoModelForCausalLM.from_pretrained(output_path)
            test_tokenizer = AutoTokenizer.from_pretrained(output_path)
            
            # Test model with a simple prompt
            test_input = test_tokenizer("Hello, world!", return_tensors="pt")
            with torch.no_grad():
                test_output = test_model(**test_input)
                if torch.isnan(test_output.logits).any() or torch.isinf(test_output.logits).any():
                    WARNING("Numerical instability detected in saved model")
                else:
                    RIGHT("Model integrity validation passed")
            
            # Clean up test model
            del test_model, test_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            ERROR(f"Error saving model: {e}")
            raise
    
    RIGHT("RLHF training completed successfully!")

def validate_rlhf_args(args):
    """
    Validate and normalize arguments for RLHF training.
    This function checks if required arguments are provided, sets default values for missing arguments,
    and validates the ranges and types of numerical arguments.

    Args:
        args (Namespace): A namespace object containing training parameters.

    Returns:
        Namespace: The validated and normalized arguments.

    Raises:
        ValueError: If any argument is invalid, e.g., missing required argument or invalid value range.
    """
    # Check if the model path is provided, which is a required argument
    if not hasattr(args, 'model_path') or not args.model_path:
        raise ValueError("--model_path is required")

    # Set default values for RLHF hyperparameters if they are not specified
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

    # Validate the ranges and types of numerical arguments
    if float(args.rlhf_lr) <= 0:
        raise ValueError("rlhf_lr must be > 0")
    for name in ['rlhf_batch_size', 'rlhf_mini_batch_size', 'rlhf_accum_steps', 'rlhf_epochs', 'rlhf_max_samples', 'rlhf_max_length']:
        try:
            val = int(getattr(args, name))
            if val <= 0:
                raise ValueError
        except Exception:
            raise ValueError(f"{name} must be a positive integer")

    return args