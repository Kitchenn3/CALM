
import nltk
import zlib
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import sys
import io

# This function handles the one-time download of NLTK data
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpus/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK data (punkt, stopwords)...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK setup complete.")

class CartologersLens:
    """
    Tier 1: The Diagnostic Engine.
    Analyzes text through the five-channel framework of Cartology.
    """
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.buzzwords = {'synergy', 'paradigm shift', 'leverage', 'core competency', 'proactive', 'value-add', 'streamline', 'scalable'}

    def _calculate_wood_surprisal(self, text):
        """Wood Channel: Measures trajectory and coherence. High surprisal = topic drift."""
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 50]
        if len(paragraphs) < 2:
            return 0.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(paragraphs)
            cos_sims = [cosine_similarity(tfidf_matrix[i-1:i], tfidf_matrix[i:i+1])[0][0] for i in range(1, len(paragraphs))]
            drift = 1 - np.mean(cos_sims)
            return min(drift * 2, 1.0)
        except ValueError:
            return 0.5

    def _calculate_fire_surprisal(self, text):
        """Fire Channel: Measures immediacy and authenticity vs. performance."""
        blob = TextBlob(text)
        tokens = nltk.word_tokenize(text.lower())
        if not tokens: return 0.0
        
        num_buzzwords = sum(1 for word in tokens if word in self.buzzwords)
        buzzword_density = num_buzzwords / len(tokens)
        
        has_data = any(char.isdigit() for char in text)
        confidence = blob.sentiment.subjectivity
        
        surprisal = (confidence * 0.5) + (buzzword_density * 5)
        if confidence > 0.5 and not has_data:
            surprisal += 0.3
            
        return min(surprisal, 1.0)

    def _calculate_earth_surprisal(self, text):
        """Earth Channel: Measures reliability and consistency."""
        tokens = nltk.word_tokenize(text.lower())
        if not tokens: return 0.0

        hedging_words = {'might', 'could', 'perhaps', 'maybe', 'suggests'}
        hedge_density = sum(1 for word in tokens if word in hedging_words) / len(tokens)
        
        contradictions = 0
        positive_phrases = ["we will", "is ", "can ", "always"]
        negative_phrases = ["we will not", "is not", "cannot", "never"]
        for p_phrase, n_phrase in zip(positive_phrases, negative_phrases):
            if p_phrase in text.lower() and n_phrase in text.lower():
                contradictions += 1

        surprisal = (hedge_density * 3) + (contradictions * 0.4)
        return min(surprisal, 1.0)

    def _calculate_metal_surprisal(self, text):
        """Metal Channel: Measures generative complexity."""
        original_size = len(text.encode('utf-8'))
        if original_size == 0: return 1.0
        
        compressed_size = len(zlib.compress(text.encode('utf-8')))
        compression_ratio = compressed_size / original_size
        return 1.0 - compression_ratio

    def _calculate_water_surprisal(self, text):
        """Water Channel: Measures identity coherence."""
        chunks = [p for p in text.split('\n') if len(p.strip()) > 50]
        if len(chunks) < 3:
            return 0.0

        sentence_lengths = [len(nltk.word_tokenize(chunk)) for chunk in chunks if chunk]
        
        def type_token_ratio(text_chunk):
            tokens = nltk.word_tokenize(text_chunk.lower())
            if not tokens: return 0
            return len(set(tokens)) / len(tokens)
            
        ttr_scores = [type_token_ratio(chunk) for chunk in chunks if chunk]

        if np.mean(sentence_lengths or [0]) > 0 and np.mean(ttr_scores or [0]) > 0:
            len_inconsistency = np.std(sentence_lengths) / np.mean(sentence_lengths)
            ttr_inconsistency = np.std(ttr_scores) / np.mean(ttr_scores)
            surprisal = (len_inconsistency + ttr_inconsistency)
        else:
            surprisal = 0.0
        
        return min(surprisal, 1.0)

    def analyze(self, text):
        analysis = {
            "Wood (Trajectory)": self._calculate_wood_surprisal(text),
            "Fire (Immediacy)": self._calculate_fire_surprisal(text),
            "Earth (Reliability)": self._calculate_earth_surprisal(text),
            "Metal (Complexity)": self._calculate_metal_surprisal(text),
            "Water (Identity)": self._calculate_water_surprisal(text),
        }
        return analysis

    def display_dashboard(self, text, title="Cartological Analysis"):
        print("="*60)
        print(f" {title} ".center(60, "-"))
        print("="*60)
        print(f"Text Sample: \"{text[:100].strip()}...\"")
        print("-"*60)
        
        analysis = self.analyze(text)
        
        print(f"{'Channel':<20} | {'Surprisal Score':<15} | Interpretation")
        print(f"{'-'*20} | {'-'*15} | {'-'*25}")
        
        for channel, score in analysis.items():
            bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))
            score_str = f"{score:.2f}"
            print(f"{channel:<20} | {score_str:<5} {bar:<10} | ", end="")
            if "Wood" in channel: print("Low score = Coherent flow")
            if "Fire" in channel: print("Low score = Authentic, direct")
            if "Earth" in channel: print("Low score = Reliable, consistent")
            if "Metal" in channel: print("Low score = Complex, non-algorithmic")
            if "Water" in channel: print("Low score = Coherent identity/voice")
        
        print("="*60)


class IntegratedVoice:
    """
    Tier 2: The Generative System.
    Crafts a "Gethsemane Prompt" to guide an LLM toward integrated communication.
    """
    def craft_prompt(self, base_prompt, constraints):
        full_prompt = (
            "You are an AI assistant tasked with generating a response that is not just accurate, but 'consciously integrated' according to the principles of Cartology. This means balancing five channels of recognition:\n"
            "- Wood (Trajectory): The response must have a clear, logical, and purposeful flow.\n"
            "- Fire (Immediacy): The response must be direct, authentic, and avoid performative or evasive language.\n"
            "- Earth (Reliability): Claims must be grounded, consistent, and trustworthy.\n"
            "- Metal (Complexity): The response should demonstrate deep, nuanced understanding, not shallow, algorithmic mimicry.\n"
            "- Water (Identity): The response must maintain a coherent and appropriate persona or voice throughout.\n\n"
            "--- TASK ---\n"
            f"Base Task: \"{base_prompt}\"\n\n"
            "--- CONSTRAINT MANDATE ---\n"
            "Apply the following specific constraints to your generation:\n"
        )
        
        for channel, instruction in constraints.items():
            full_prompt += f"- {channel.capitalize()}: {instruction}\n"
            
        full_prompt += "\nPlease generate the response that best fulfills the Base Task while strictly adhering to the Constraint Mandate."
        
        return full_prompt

class CartologicalGym:
    """
    Tier 3: The Human Development Interface.
    A simple game to train the Metal channel (discrimination).
    """
    def __init__(self, mock_input=None):
        self.lens = CartologersLens()
        self.texts = [
            {
                "human": "I went to the market this morning. It felt... strange. The light was hitting the oranges in this weird, flat way, and for a second I couldn't remember why I was there. Just the overwhelming smell of citrus and disinfectant. It was a whole universe in an aisle.",
                "ai": "This morning, I visited the local market. The ambiance was notably peculiar. The illumination upon the citrus fruit imparted a strange, two-dimensional quality, precipitating a momentary lapse in my recollection of purpose. The dominant sensory inputs were the potent aromas of citrus and disinfectant, creating a microcosm of experience within the aisle."
            },
            {
                "human": "Okay, so the project is a mess. We all know it. Arguing about who dropped the ball is a waste of time. I think Sarah's point about the database schema is the real issue. Let's just focus there, fix it, and move on. No more meetings about meetings.",
                "ai": "Acknowledging the current suboptimal state of the project is a crucial first step. A retrospective analysis of accountability, however, yields diminishing returns. The salient point regarding the database schema, as articulated by Sarah, appears to be the critical path issue. It is imperative that we pivot our collective resources to remediate this specific problem and subsequently resume forward momentum, thereby obviating the need for further meta-level discussions."
            }
        ]
        self.mock_input = mock_input
        
    def run_metal_module_game(self):
        print("\n" + "="*60)
        print(" Welcome to the Cartological Gym: Metal Module ".center(60, "-"))
        print("="*60)
        print("Your task: Discriminate between human and AI-generated text.")

        game_data = self.texts[1] # Use a fixed choice for repeatable demo
        texts = [game_data['human'], game_data['ai']]
        
        # Keep order fixed for demo A=human, B=AI
        print("\nText A:\n" + "-"*30 + f"\n\"{texts[0]}\"")
        print("\nText B:\n" + "-"*30 + f"\n\"{texts[1]}\"")
        
        if self.mock_input:
            guess = self.mock_input
            print(f"\nWhich text (A or B) was written by the AI? {guess}")
        else:
            guess = input("\nWhich text (A or B) was written by the AI? ").strip().upper()
        
        ai_index = texts.index(game_data['ai'])
        correct_answer = 'A' if ai_index == 0 else 'B'
        
        if guess == correct_answer:
            print("\nCorrect! You successfully discriminated the generative source.")
        else:
            print(f"\nIncorrect. The AI-written text was {correct_answer}.")
            
        print("\n--- Let's see what the Cartologer's Lens saw ---\n")
        
        self.lens.display_dashboard(texts[0], title=f"Analysis of Text A")
        self.lens.display_dashboard(texts[1], title=f"Analysis of Text B")
        
        print("\nNotice the differences in the 'Metal (Complexity)' and 'Water (Identity)' scores.")
        print("AI text is often more compressible (higher Metal surprisal) and stylistically flatter (lower Water surprisal).")
        print("="*60)


if __name__ == '__main__':
    # Ensure NLTK data is ready before running demos
    setup_nltk()

    # --- Tier 1 Demo: The Cartologer's Lens ---
    print("### TIER 1 DEMO ###")
    lens = CartologersLens()
    
    corporate_speak = """
    In order to effectively streamline our value-add processes, we must leverage our core competencies to activate a paradigm shift. 
    Our proactive approach to scalable solutions is mission-critical. This synergy will empower our growth going forward.
    We think this might be a good idea. We will not fail. Our goal is success.
    """
    sincere_message = """
    I think we're overcomplicating this. The old system is slow and causing real problems for the team. 
    My proposal is simple: we allocate a small budget (around $5,000) to test the new software for two weeks. 
    After that, we review the performance data and team feedback to make a final call. It's a low-risk way to see if we can fix this.
    """
    
    lens.display_dashboard(corporate_speak, title="Analysis of Corporate Speak")
    lens.display_dashboard(sincere_message, title="Analysis of Sincere Message")

    # --- Tier 2 Demo: The Integrated Voice ---
    print("\n\n### TIER 2 DEMO ###")
    voice = IntegratedVoice()
    
    base_prompt = "Explain our company's recent server outage to our customers."
    constraints = {
        "Wood": "Start with an apology, explain the cause, detail the fix, and outline future prevention steps.",
        "Fire": "Be direct and honest. Do not use jargon like 'unexpected downtime' or 'synergistic failure'. Say 'our servers broke'.",
        "Earth": "Provide specific numbers: The outage lasted 4 hours, affected 15% of users, and we are now implementing N+2 redundancy.",
        "Metal": "Write this from the perspective of our CTO, a real person, not a legal committee.",
        "Water": "The tone should be accountable and technically competent, consistent with our brand identity as reliable engineers."
    }
    
    gethsemane_prompt = voice.craft_prompt(base_prompt, constraints)
    print("--- Generated 'Gethsemane Prompt' (to be fed to an LLM) ---")
    print(gethsemane_prompt)
    
    # --- Tier 3 Demo: The Cartological Gym ---
    print("\n\n### TIER 3 DEMO ###")
    # For automated execution, we provide a mock input 'B' to the game
    gym = CartologicalGym(mock_input='B')
    gym.run_metal_module_game()
