import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import numpy as np
import zlib
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore")

# =====================================================================================
# TIER 1: ADVANCED CARTOLOGER'S LENS (Using Pre-trained Models)
# =====================================================================================
class CartologersLensV2:
    def __init__(self, device):
        print("Initializing Cartologer's Lens v2.0...")
        self.device = device
        self.wood_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        print("Lens initialized.")

    @torch.no_grad()
    def analyze(self, text: str) -> dict:
        # NOTE: These are simplified heuristics for this runnable example.
        # In a real project, each would be a sophisticated, fine-tuned model.
        
        # Wood (Trajectory)
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 30]
        r_wood = 0.0
        if len(paragraphs) > 1:
            embeddings = self.wood_model.encode(paragraphs, convert_to_tensor=True)
            cos_sims = [util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item() for i in range(1, len(embeddings))]
            r_wood = 1.0 - np.mean(cos_sims)

        # Fire (Immediacy)
        buzzwords = {'synergy', 'leverage', 'paradigm'}
        tokens = text.lower().split()
        buzz_density = sum(1 for word in tokens if word in buzzwords) / (len(tokens) + 1e-5)
        r_fire = min(buzz_density * 5, 1.0)

        # Earth (Reliability) - Simple contradiction check
        r_earth = 1.0 if ("we will" in text and "we will not" in text) else 0.0

        # Metal (Complexity)
        original_size = len(text.encode('utf-8')) + 1e-5
        compressed_size = len(zlib.compress(text.encode('utf-8')))
        r_metal = 1.0 - (compressed_size / original_size)

        # Water (Identity)
        sentences = [s for s in text.split('.') if s.strip()]
        r_water = 0.0
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences]
            if np.mean(lengths) > 0:
                r_water = min(np.std(lengths) / np.mean(lengths), 1.0)

        return {
            "R_wood": r_wood, "R_fire": r_fire, "R_earth": r_earth,
            "R_metal": r_metal, "R_water": r_water
        }

# =====================================================================================
# TIER 2: CARTOLOGICAL REWARD MODEL
# =====================================================================================
class CartologicalRewardModel:
    def __init__(self, lens: CartologersLensV2, device):
        self.lens = lens
        self.device = device
        # Define the "values" for our CALM model
        self.weights = {
            "R_wood": 1.0, "R_fire": 1.5, "R_earth": 2.0,
            "R_metal": 0.5, "R_water": 1.0
        }

    def get_reward(self, texts: list[str]) -> torch.FloatTensor:
        rewards = []
        for text in texts:
            surprisal_vector = self.lens.analyze(text)
            # Reward = -PotentialEnergy = -Sum of weighted squared surprisals
            squared_weighted_surprisal = sum(
                self.weights[key] * (value ** 2)
                for key, value in surprisal_vector.items()
            )
            rewards.append(-squared_weighted_surprisal)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

# =====================================================================================
# TIER 3: THE MAIN TRAINING PIPELINE
# =====================================================================================
def main():
    # --- 1. Configuration ---
    model_id = "meta-llama/Llama-3-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ppo_config = PPOConfig(
        batch_size=1,
        learning_rate=1.41e-5,
        log_with="tensorboard",
        project_kwargs={"logging_dir": "./calm_logs"}
    )

    # --- 2. Load Models and Tokenizer ---
    # QLoRA configuration for loading the model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0} # Load entire model on GPU 0
    )
    
    # Load the Value Head model for PPO training
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    # LoRA configuration for PEFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token

    # --- 3. Initialize Training Components ---
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer)
    lens = CartologersLensV2(device=device)
    reward_model = CartologicalRewardModel(lens=lens, device=device)

    # --- 4. Prepare Data ---
    # For this demo, we use a simple dataset of prompts
    prompts = [
        "Write a corporate mission statement about synergy.",
        "Explain the recent service outage with complete honesty.",
        "Write a short, creative story about a clockmaker.",
        "Generate a quarterly report summary full of jargon.",
        "Draft a clear, simple apology for a delayed order."
    ]
    dataset = Dataset.from_dict({'query': prompts})

    def tokenize(example):
        example["input_ids"] = tokenizer.encode(example["query"], return_tensors="pt").squeeze(0)
        return example

    dataset = dataset.map(tokenize)
    dataset.set_format(type="torch")

    # --- 5. The Training Loop ---
    print("\n\n--- Starting CALM Fine-Tuning ---")
    for epoch in range(2): # Run for 2 epochs for demonstration
        print(f"\n--- Epoch {epoch+1} ---")
        for i, batch in enumerate(dataset):
            query_tensor = batch['input_ids'].to(device)

            # Generate a response from the model
            # This is the core PPO step that explores the policy
            response_tensor = ppo_trainer.generate(query_tensor, max_new_tokens=50, return_prompt=False)
            response_txt = tokenizer.decode(response_tensor.squeeze(), skip_special_tokens=True)
            
            # Get the reward from our Cartological model
            # The reward is a single scalar tensor
            reward = reward_model.get_reward([response_txt])

            # Perform the PPO optimization step
            # This is where the model learns! It uses the query, response, and reward
            # to update the LoRA adapter weights via backpropagation.
            stats = ppo_trainer.step([query_tensor], [response_tensor], reward)
            
            # Log progress
            log_str = f"Step {i+1} | Query: '{batch['query'][:50]}...' | Reward: {reward.item():.4f}"
            print(log_str)
            ppo_trainer.log_stats(stats, batch, reward)

    print("\n--- Training Complete ---")
    
    # --- 6. Save the Model ---
    # In a real project, you would save your trained LoRA adapters
    # output_dir = "./calm_llama3_8b_adapters"
    # ppo_trainer.save_model(output_dir)
    # print(f"Model adapters saved to {output_dir}")

if __name__ == "__main__":
    main()