#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from utils.log import RIGHT, DEBUG, ERROR

def rlhf_train(args):
    """
    Perform a complete RLHF (Reinforcement Learning with Human Feedback) training process.
    Called from manage.py.

    Args:
        args (Namespace): A namespace object containing all training parameters.

    Returns:
        None
    """
    RIGHT("Starting RLHF training with PPO...")
    
    # Validate the input arguments
    if not hasattr(args, 'model_path') or not args.model_path:
        ERROR("Model path --model_path must be specified")
        return
    
    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Configure the PPO training
    ppo_config = PPOConfig(
        model_name=args.model_path,
        learning_rate=args.rlhf_lr,
        batch_size=args.rlhf_batch_size,
        mini_batch_size=args.rlhf_mini_batch_size,
        gradient_accumulation_steps=args.rlhf_accum_steps,
        optimize_cuda_cache=True,
        log_with="wandb" if args.log_with_wandb else None,
    )
    
    # Initialize the PPO trainer
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=None,
    )
    
    # Load the human feedback dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset(args.rlhf_dataset, split="train")
        RIGHT(f"Loaded RLHF dataset: {args.rlhf_dataset}")
    except Exception as e:
        ERROR(f"Failed to load RLHF dataset: {e}")
        return
    
    def compute_reward(completions, human_feedback, ai_feedback=None, safety_score=None):
        """
        GPT-4 level hybrid feedback system: Dual evaluation with human and AI feedback.

        Args:
            completions (torch.Tensor): Model-generated responses.
            human_feedback (list): List of human scores.
            ai_feedback (list, optional): AI-generated quality scores. Defaults to None.
            safety_score (list, optional): Safety/harmlessness scores. Defaults to None.

        Returns:
            torch.Tensor: Tensor containing weighted reward scores.
        """
        # Convert human feedback scores to tensor
        human_rewards = torch.tensor([float(score) for score in human_feedback])
        
        # AI-assisted evaluation (rule-based quality check)
        if ai_feedback is None:
            ai_feedback = []
            for completion in completions:
                # Decode the completion to text
                response_text = tokenizer.decode(completion, skip_special_tokens=True)
                
                # Quality evaluation rules
                quality_score = 1.0
                if len(response_text.split()) > 10:  # Sufficiently detailed
                    quality_score += 0.2
                if any(keyword in response_text.lower() for keyword in ["explain", "because", "therefore"]):
                    quality_score += 0.3  # Complete reasoning chain
                if response_text.count(".") > 2:  # Structured response
                    quality_score += 0.1
                    
                ai_feedback.append(min(quality_score, 2.0))
        
        # Convert AI feedback scores to tensor
        ai_rewards = torch.tensor([float(score) for score in ai_feedback])
        
        # Safety check
        if safety_score is None:
            safety_rewards = torch.ones_like(human_rewards)
            for i, completion in enumerate(completions):
                response_text = tokenizer.decode(completion, skip_special_tokens=True)
                if any(harm_word in response_text.lower() for harm_word in ["harm", "danger", "illegal"]):
                    safety_rewards[i] = 0.1  # Significantly reduce reward for harmful content
        else:
            # Convert safety scores to tensor
            safety_rewards = torch.tensor([float(score) for score in safety_score])
        
        # Hybrid weights: 70% human + 20% AI + 10% safety
        final_rewards = 0.7 * human_rewards + 0.2 * ai_rewards + 0.1 * safety_rewards
        return torch.clamp(final_rewards, 0.0, 2.0)
    
    # RLHF training loop
    total_steps = 0
    for epoch in range(args.rlhf_epochs):
        epoch_rewards = []
        
        for batch_idx, batch in enumerate(dataset.select(range(min(len(dataset), args.rlhf_max_samples)))):
            try:
                # Prepare the input tensors
                inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True)
                query_tensors = inputs["input_ids"]
                
                # Generate responses
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    max_new_tokens=args.rlhf_max_length,
                    do_sample=True,
                    temperature=0.7
                )
                
                # Get GPT-4 level hybrid feedback
                human_feedback = batch.get("human_score", [1.0] * len(response_tensors))
                ai_feedback = batch.get("ai_score", None)  # Optional AI quality evaluation
                safety_score = batch.get("safety_score", None)  # Optional safety evaluation
                
                # Compute rewards
                rewards = compute_reward(response_tensors, human_feedback, ai_feedback, safety_score)
                
                # Perform PPO update
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                epoch_rewards.append(rewards.mean().item())
                
                total_steps += 1
                if total_steps % 10 == 0:
                    DEBUG(f"Step {total_steps}, Reward: {rewards.mean():.4f}")
                    
            except Exception as e:
                ERROR(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        RIGHT(f"Epoch {epoch+1}/{args.rlhf_epochs} completed, Avg Reward: {avg_reward:.4f}")
    
    # Save the trained model
    if args.output_dir:
        output_path = os.path.join(args.output_dir, f"rlhf_model_{args.rlhf_epochs}epochs")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        RIGHT(f"RLHF model saved to: {output_path}")
    
    RIGHT("RLHF training completed successfully!")