
### **Prototype v0.0002**
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import numpy as np
import zlib
from sentence_transformers import SentenceTransformer, util
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =====================================================================================
# TIER 1 UPGRADE: CartologersLens v3 - With Model-Based Earth Channel & Batching
# =====================================================================================
class CartologersLensV3:
    def __init__(self, device):
        print("Initializing Cartologer's Lens v3.0...")
        self.device = device
        
        # Wood Channel Model
        self.wood_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Earth Channel Model (UPGRADE)
        print("Loading Earth Channel (NLI) model... This may take a moment.")
        self.earth_model_pipe = pipeline(
            "text-classification", 
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", 
            device=0 if self.device == "cuda" else -1 # Use GPU if available
        )
        print("Lens initialized.")

    @torch.no_grad()
    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyzes a BATCH of texts for improved performance."""
        batch_size = len(texts)
        results = [{} for _ in range(batch_size)]

        # --- Wood Channel (Batch) ---
        r_wood_scores = self._analyze_wood_batch(texts)
        for i in range(batch_size): results[i]['R_wood'] = r_wood_scores[i]

        # --- Earth Channel (Batch) ---
        r_earth_scores = self._analyze_earth_batch(texts)
        for i in range(batch_size): results[i]['R_earth'] = r_earth_scores[i]
        
        # --- Other Channels (Analyzed individually for this version) ---
        for i, text in enumerate(texts):
            results[i]['R_fire'] = self._analyze_fire_single(text)
            results[i]['R_metal'] = self._analyze_metal_single(text)
            results[i]['R_water'] = self._analyze_water_single(text)
            
        return results

    def _analyze_wood_batch(self, texts: list[str]) -> list[float]:
        scores = []
        for text in texts:
            paragraphs = [p for p in text.split('\n') if len(p.strip()) > 30]
            if len(paragraphs) < 2:
                scores.append(0.0)
                continue
            embeddings = self.wood_model.encode(paragraphs, convert_to_tensor=True, show_progress_bar=False)
            cos_sims = [util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item() for i in range(1, len(embeddings))]
            scores.append(1.0 - np.mean(cos_sims))
        return scores

    def _analyze_earth_batch(self, texts: list[str]) -> list[float]:
        """UPGRADED: Uses a real NLI model to detect contradictions."""
        scores = []
        for text in texts:
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            if len(sentences) < 2:
                scores.append(0.0)
                continue
            
            sentence_pairs = []
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    sentence_pairs.append(f"{sentences[i]}[SEP]{sentences[j]}")

            if not sentence_pairs:
                scores.append(0.0)
                continue

            # NLI model predicts contradiction probability for all pairs at once
            predictions = self.earth_model_pipe(sentence_pairs, batch_size=8) # Use internal batching
            
            max_contradiction_score = 0.0
            for pred in predictions:
                if isinstance(pred, list): # The pipeline might return a list of lists
                    pred = pred[0]
                if pred['label'] == 'contradiction':
                    max_contradiction_score = max(max_contradiction_score, pred['score'])
            scores.append(max_contradiction_score)
        return scores
    
    # --- Single-instance heuristic methods (placeholders for future model upgrades) ---
    def _analyze_fire_single(self, text: str) -> float:
        buzzwords = {'synergy', 'leverage', 'paradigm'}
        tokens = text.lower().split()
        buzz_density = sum(1 for word in tokens if word in buzzwords) / (len(tokens) + 1e-5)
        return min(buzz_density * 5, 1.0)

    def _analyze_metal_single(self, text: str) -> float:
        original_size = len(text.encode('utf-8')) + 1e-5
        compressed_size = len(zlib.compress(text.encode('utf-8')))
        return 1.0 - (compressed_size / original_size)

    def _analyze_water_single(self, text: str) -> float:
        sentences = [s for s in text.split('.') if s.strip()]
        if len(sentences) < 3: return 0.0
        lengths = [len(s.split()) for s in sentences]
        if np.mean(lengths) == 0: return 0.0
        return min(np.std(lengths) / np.mean(lengths), 1.0)


# =====================================================================================
# TIER 2 UPGRADE: CartologicalRewardModel v2 - With Reward Normalization
# =====================================================================================
class CartologicalRewardModelV2:
    def __init__(self, lens: CartologersLensV3, device):
        self.lens = lens
        self.device = device
        self.weights = {
            "R_wood": 1.0, "R_fire": 1.5, "R_earth": 2.0,
            "R_metal": 0.5, "R_water": 1.0
        }
        # For reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_buffer = []

    def get_reward(self, texts: list[str]) -> torch.FloatTensor:
        surprisal_vectors = self.lens.analyze_batch(texts)
        
        rewards = []
        for surprisal_vector in surprisal_vectors:
            squared_weighted_surprisal = sum(
                self.weights[key] * (value ** 2)
                for key, value in surprisal_vector.items()
            )
            rewards.append(-squared_weighted_surprisal)
        
        # --- REWARD NORMALIZATION (UPGRADE) ---
        self.reward_buffer.extend(rewards)
        if len(self.reward_buffer) > 100: # Keep a running buffer of last 100 rewards
            self.reward_buffer = self.reward_buffer[-100:]
            self.reward_mean = np.mean(self.reward_buffer)
            self.reward_std = np.std(self.reward_buffer)

        normalized_rewards = [(r - self.reward_mean) / (self.reward_std + 1e-5) for r in rewards]
        
        return torch.tensor(normalized_rewards, dtype=torch.float32, device=self.device)


# =====================================================================================
# TIER 3 UPGRADE: THE MAIN TRAINING PIPELINE v2 - With Real Dataset & Batching
# =====================================================================================
def main():
    # --- 1. Configuration (UPGRADE) ---
    model_id = "meta-llama/Llama-3-8B-Instruct"
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ppo_config = PPOConfig(
        batch_size=4,  # Increased batch size
        mini_batch_size=2,
        gradient_accumulation_steps=2, # Effective batch size = 4*2 = 8
        learning_rate=1.41e-5,
        log_with="tensorboard", # Switch to "wandb" if you have it installed
        project_kwargs={"logging_dir": "./calm_logs_v2"}
    )

    # --- 2. Load Models and Tokenizer (Same as v1) ---
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Initialize Training Components ---
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer)
    lens = CartologersLensV3(device=device)
    reward_model = CartologicalRewardModelV2(lens=lens, device=device)

    # --- 4. Prepare Data (UPGRADE) ---
    print("Loading and preparing dataset...")
    # Load a small part of a real dataset for this demo
    dataset = load_dataset(dataset_name, split="train_sft[:2%]").shuffle(seed=42).select(range(100))
    
    def format_prompt(example):
        # The dataset has a 'messages' column with conversations. We'll use the user's first message as the query.
        prompt = example['messages'][0]['content']
        # We need a consistent formatting for the model
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        example['query'] = formatted_prompt
        example['input_ids'] = tokenizer.encode(example['query'])
        return example

    dataset = dataset.map(format_prompt)
    dataset.set_format(type="torch")

    # --- 5. The Training Loop ---
    print("\n\n--- Starting CALM Fine-Tuning v2 ---")
    for epoch in range(1): # One epoch over the small dataset is enough for a demo
        print(f"\n--- Epoch {epoch+1} ---")
        for batch in tqdm(ppo_trainer.pt_dataloader(dataset)):
            query_tensors = batch['input_ids']

            # Generate responses in batch
            response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=60, return_prompt=False)
            
            # Decode responses for reward calculation
            response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Get reward in batch
            reward_tensors = reward_model.get_reward(response_texts)
            
            # PPO optimization step
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            ppo_trainer.log_stats(stats, batch, reward_tensors)

    print("\n--- Training Complete ---")
    
    # --- 6. Save the Model ---
    output_dir = "./calm_llama3_8b_v2_adapters"
    ppo_trainer.save_model(output_dir)
    print(f"Model adapters saved to {output_dir}")

if __name__ == "__main__":
    main()