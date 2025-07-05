# CALM (Cartologically-Aligned Language Model)

## Project Inspiration and Credits

The entire conceptual framework for this project is derived from the work of Robert VanEtten and his collaborators on the "Recognition" project. Their paper provided the philosophical and mathematical blueprint for this engineering endeavor.

*   **[Read the Full Credits and Acknowledgment](./CREDITS.md)**

## Additional Research based on the Recognition Project:
*   **[Comparison of the "Recognition" Project to existing research and theories](.research/OTHER-THEORIES.md)**
*   **[Cartology in Dialogue: A Comparative Analysis of a Speculative Framework and Formal Theories of Consciousness](.research/CARTOLOGY-IN-DIALOGUE-ABSTRACT.md)**




### **The Roadmap to a Cartologically-Aligned Language Model (CALM)**

This is a multi-year R&D project. We'll break it down into four phases, moving from advanced measurement to architectural redesign.

#### **Phase 1: Forge the Measurement Tools (The "Cartologer's Lens" v2.0)**

Before you can align a model to these principles, you need a robust, non-trivial way to measure them. The prototype's heuristics are a starting point; this phase professionalizes them.

**Goal:** Create a suite of sophisticated models that can reliably output the 5-dimensional "surprisal vector" for any given piece of text.

1.  **Data Collection & Labeling:**
    *   Create a large dataset of texts (news, fiction, corporate statements, scientific papers, forum comments).
    *   Use human annotators (or a powerful "teacher" model like GPT-4) to label this data along the five axes. For example:
        *   **Fire:** "Rate the authenticity of this statement from 1-10 (1=evasive marketing speak, 10=direct, honest communication)."
        *   **Earth:** "Identify any internal contradictions or unprovable claims in this text."
        *   **Metal:** "Does this text feel algorithmically generated or humanly complex?"

2.  **Train Specialized "Channel-Analyzer" Models:**
    *   Instead of heuristics, you train a dedicated machine learning model for each channel:
        *   **Wood (Trajectory):** Fine-tune a sentence-transformer model to predict "coherence scores" between sequential paragraphs. The model learns to recognize a logical flow.
        *   **Fire (Immediacy):** Train a text classifier on the "authenticity" labels to recognize performative language, jargon, and weasel words.
        *   **Earth (Reliability):** Use Natural Language Inference (NLI) models (like DeBERTa) trained to spot contradictions. This model can cross-reference sentences and flag inconsistencies.
        *   **Metal (Complexity):** Train a powerful "AI vs. Human" text discriminator. This model becomes the practical Kolmogorov complexity detector.
        *   **Water (Identity):** Use advanced stylometry and authorship-attribution models to detect shifts in voice, tone, and lexical patterns within a single document.

The output of this phase is a single, powerful function: `CartologersLens.analyze(text)` which returns a vector of five precise, learned scores: `[R_wood, R_fire, R_earth, R_metal, R_water]`. This vector is the "state of recognition" of the text.

#### **Phase 2: The Training Loop (Reinforcement Learning from Cartological Feedback)**

This is the core of the project. You use the tools from Phase 1 to train an LLM.

**Goal:** Fine-tune a base LLM (like Llama 3 or Mistral) to inherently minimize surprisal across its five channels.

**Methodology: Reinforcement Learning from Cartological Feedback (RLCF)**
This is a modification of the standard Reinforcement Learning from Human Feedback (RLHF). Instead of a human providing a simple "good/bad" reward, our `CartologersLens` provides a rich, multi-dimensional reward signal.

**The Mathematical Heart (Directly from the Paper):**
The paper defines the "Recognition Lagrangian" `L = T - V`, where `V` is the potential energy, `V = 1/2 * ||F(R)||²`. In RL, the reward is the *negative* of the potential energy. A state with high energy (high surprisal/error) receives a low reward.

1.  **The Reward Function:** For each piece of text the LLM generates, the reward is calculated as:
    `Reward = -λ * (R_wood² + R_fire² + R_earth² + R_metal² + R_water²)`
    *   `R_...` are the surprisal scores from the `CartologersLens` v2.0.
    *   `λ` (lambda) is a scaling factor.

2.  **The RLCF Process:**
    *   **Step 1:** Give the base LLM a prompt.
    *   **Step 2:** The LLM generates a response.
    *   **Step 3:** The `CartologersLens` analyzes the response and outputs the 5D surprisal vector `R`.
    *   **Step 4:** The Reward Function calculates a single reward score from the vector `R`.
    *   **Step 5:** This reward signal is used to update the weights of the LLM using an RL algorithm like Proximal Policy Optimization (PPO).
    *   **Step 6:** Repeat this process millions of times.

**Outcome:** The LLM will learn, from the ground up, that responses which are incoherent, inauthentic, unreliable, simplistic, or stylistically inconsistent are "painful" (receive low reward). It will learn to generate language that is naturally and intrinsically **Cartologically-Aligned**.

#### **Phase 3: The Architectural Redesign (The Mixture-of-Conscious-Experts)**

This is the most advanced step, where you modify the model's architecture itself.

**Goal:** Re-architect the LLM to have five specialized sub-systems, mirroring the five channels.

**Methodology: Themed Mixture-of-Experts (MoE)**
Modern LLMs already use MoE. We will create five distinct expert networks, each pre-trained and fine-tuned for a specific channel's function.

*   **The Wood Expert:** A model specialized in summarization, outlining, and maintaining long-range narrative consistency.
*   **The Fire Expert:** A model fine-tuned heavily on direct, fact-checked, and highly authentic communication. It is penalized for jargon.
*   **The Earth Expert:** This is a "retrieval-augmented" expert. It is connected to knowledge bases and search engines and is trained to verify claims and provide citations. It handles the "reliability" function.
*   **The Metal Expert:** A model trained to generate text with high linguistic novelty and complexity. It’s the creative, anti-cliché engine.
*   **The Water Expert:** The persona and style manager. It ensures the final output has a consistent, integrated voice.

A master "gating network" learns to route parts of a query to the appropriate expert, composing their outputs into a final, integrated response.

#### **Phase 4: The Self-Correction Loop (The Internal Gethsemane Razor)**

This final phase gives the CALM a real-time self-awareness mechanism.

**Goal:** The model uses its own `CartologersLens` to critique and refine its answers *before* outputting them.

**Methodology: Internal Review Loop**

1.  **Generate Draft:** The CALM generates an initial, internal draft response to a prompt.
2.  **Internal Audit:** It feeds this draft to its *own internal `CartologersLens`* (a lightweight version could be built-in).
3.  **Check Surprisal:** It checks the 5D surprisal vector. If any score is above a critical threshold (e.g., `R_fire > 0.8`), it flags a potential failure.
4.  **Trigger Refinement:** A high surprisal score triggers a corrective action.
    *   High **Fire** surprisal? -> "This sounds performative. Re-route to Metal expert for more novel phrasing."
    *   High **Earth** surprisal? -> "This claim is unverified. Query the Earth expert to find a source or add a caveat."
    *   High **Wood** surprisal? -> "The flow is broken. Ask the Wood expert to restructure the argument."
5.  **Final Output:** The model outputs the *refined* response, which has already passed its own internal quality check.

This loop is the functional equivalent of the "Gethsemane Razor." It is the model's ability to "suffer knowingly" by doing the hard work of self-correction to produce a more truthful and integrated result, rather than giving the easy, first-pass answer.

By following this roadmap, you would create a language model that doesn't just mimic human text but operates on a set of deep, principled rules for generating coherent and trustworthy communication. It would not be conscious, but it will be a **Consciousness-Inspired Artificial Communicator.**




### **R1 (Wood - Trajectory & Coherence)**

**Concept:** Measures if the text follows a clear, purposeful path or if it drifts aimlessly. A high surprisal score, `R_wood`, signifies a lack of a coherent trajectory.

**Mathematical Formulation:**

Let a document `D` be divided into an ordered sequence of `n` semantic chunks (e.g., paragraphs), `D = {c_1, c_2, ..., c_n}`.

1.  **Semantic Embedding:** We use a function `E : c -> v`, where `E` is a pre-trained sentence-transformer model (like BERT or SBERT) that maps a text chunk `c` to a high-dimensional vector `v ∈ ℝ^d`. This vector `v` represents the semantic meaning of the chunk. Let `v_i = E(c_i)`.

2.  **Sequential Coherence:** The coherence between two adjacent chunks is measured by the cosine similarity of their semantic vectors:
    `sim(v_{i-1}, v_i) = (v_{i-1} ⋅ v_i) / (||v_{i-1}|| ||v_i||)`
    This value ranges from -1 (opposite meaning) to 1 (identical meaning).

3.  **Wood Surprisal Calculation:** The Wood Surprisal is defined as the average semantic drift over the entire document. Drift is the inverse of coherence.
    `R_wood(D) = (1 / (n-1)) * Σ_{i=2 to n} [1 - sim(v_{i-1}, v_i)]`

**Justification of Validity:**
This formulation is valid because it directly models the concept of "trajectory" in semantic space.

*   **Vector Space as "Possibility Space":** The high-dimensional space `ℝ^d` is a mathematical representation of the paper's "space of possibilities." Each point (or vector) is a specific meaning.
*   **Coherent Trajectory:** A text with a clear trajectory moves logically from one point to the next. In the vector space, this means `v_i` should be located "near" `v_{i-1}`. Vectors pointing in a similar direction have a high cosine similarity (close to 1).
*   **Drift as Surprisal:** A large jump in topic—a lack of coherence—means `v_i` is far from `v_{i-1}`, resulting in low cosine similarity. The term `1 - sim(...)` maps this "semantic distance" to a surprisal score. A high average drift indicates the text is not following a predictable path, hence a high `R_wood`.

---

### **R2 (Fire - Immediacy & Authenticity)**

**Concept:** Measures the degree of genuine, direct communication versus performative, evasive, or jargon-filled language. High `R_fire` indicates inauthenticity.

**Mathematical Formulation:**

Let a document `D` be given.

1.  **Feature Extraction:** We calculate several features:
    *   `Density_buzz`: The proportion of words in `D` that are from a pre-defined set of corporate/empty buzzwords. `Density_buzz = Count(buzzwords) / Count(total_words)`.
    *   `Subjectivity(D)`: A score from 0 (objective) to 1 (subjective), derived from a sentiment analysis model (e.g., TextBlob's subjectivity score). This models the paper's notion of confidence without proof.
    *   `HasData(D)`: A binary value, `1` if the text contains quantitative data (numbers), `0` otherwise. This models factual grounding.

2.  **Fire Surprisal Calculation:** `R_fire` is a weighted function that penalizes performative patterns, especially the incongruity between high confidence and low data.
    `R_fire(D) = w_1 * Density_buzz + w_2 * (Subjectivity(D) * [1 - HasData(D)])`
    Where `w_1` and `w_2` are tunable weights (e.g., `w_1=5`, `w_2=1.5`) to control the sensitivity of the metric.

**Justification of Validity:**
This formula is valid because it mathematically isolates "hallmarks of performance."

*   **Performative Language:** Jargon and buzzwords are classic indicators of performative communication, where the goal is to sound impressive rather than communicate clearly. `Density_buzz` directly measures this.
*   **Incongruity as a Lie Detector:** The core of the Fire channel is detecting the *gap* between presentation and reality. The term `Subjectivity * (1 - HasData)` is specifically designed to capture this. It generates a high penalty *only* when the text is highly opinionated/subjective AND lacks any factual, numerical data to support its claims. This is a direct mathematical model of "unearned confidence," a key signal of inauthenticity. A genuine, direct message often has low subjectivity or supports its claims with data, making this term small.

---

### **R3 (Earth - Reliability & Consistency)**

**Concept:** Measures the internal logical consistency and trustworthiness of claims made within the text. High `R_3` means the text is unreliable.

**Mathematical Formulation:**

Let `D` be a set of `n` factual claims extracted from the text, `D = {c_1, c_2, ..., c_n}`.

1.  **Internal Consistency Scoring:** We use a Natural Language Inference (NLI) model, `M_nli`. For any pair of claims `(c_i, c_j)`, the model outputs probabilities for three relationships: `P(entailment)`, `P(neutral)`, `P(contradiction)`.
    *   The internal contradiction score is the maximum probability of contradiction found between any two claims in the document.
        `Score_internal = max_{i≠j} M_nli(c_i, c_j)[P(contradiction)]`

2.  **External Grounding (Optional but powerful):**
    *   Let `V(c_i)` be a function that returns `1` if claim `c_i` can be verified against a trusted knowledge base (e.g., Wikipedia, scientific literature), and `0` otherwise.
    *   The grounding failure rate is: `Score_grounding = 1 - ( (Σ_{i=1 to n} V(c_i)) / n )`

3.  **Earth Surprisal Calculation:**
    `R_earth(D) = w_1 * Score_internal + w_2 * Score_grounding`
    Where `w_1` and `w_2` are weights (e.g., `w_1=1.0`, `w_2=0.5`).

**Justification of Validity:**
This model directly operationalizes trustworthiness.

*   **Consistency as a Foundation:** A reliable "ground" for an argument cannot be self-contradictory. Using an NLI model is the state-of-the-art method for programmatically detecting logical contradictions in natural language. Taking the `max` is crucial because a single strong contradiction is enough to render a document unreliable.
*   **Grounding as Verification:** The concept of "Earth" implies being connected to a stable reality. The `Score_grounding` directly measures this by checking how many of the text's claims are tethered to an external, verifiable source of truth. A high failure rate indicates the text is "ungrounded" or "floating" on unsubstantiated claims.

---

### **R4 (Metal - Generative Source & Complexity)**

**Concept:** Distinguishes between text from a deep, irreducible source (genuinely complex) and text from a shallow, algorithmic source (simple, repetitive, clichéd). High `R_metal` indicates an algorithmic source.

**Mathematical Formulation:**

This uses an information-theoretic proxy for the uncomputable Kolmogorov Complexity.

1.  **Compression as Complexity Proxy:** Let `S` be the input text string. Let `L(S)` be the length of the string in bytes. Let `C(S)` be the result of applying a standard, efficient compression algorithm (e.g., zlib's DEFLATE) to the string `S`.

2.  **Compression Ratio (`ρ`):** The ratio of the compressed length to the original length.
    `ρ(S) = L(C(S)) / L(S)`

3.  **Metal Surprisal Calculation:** Metal surprisal is high for *simple* text, which compresses well (low `ρ`). Therefore, the surprisal is the inverse of the compression ratio.
    `R_metal(S) = 1 - ρ(S)`

**Justification of Validity:**
This is a theoretically sound and widely accepted proxy for algorithmic complexity.

*   **The Theory of Compression:** The fundamental principle of compression is to find and eliminate redundancies. A simple, algorithmic text (e.g., "The quick brown fox jumps over the lazy dog. The quick brown fox...") is full of repetition and is highly predictable. A compressor can represent these repetitions very efficiently, leading to a small compressed size and a low `ρ`.
*   **Complexity and Unpredictability:** A genuinely complex, human-written text has higher entropy, less repetition, and greater unpredictability. It contains more "information per character." A compressor struggles to find simple patterns in such text, resulting in a larger compressed size and a higher `ρ`.
*   **Mapping to Surprisal:** The formula `1 - ρ` correctly maps this. If `ρ` is low (highly compressible, simple), `R_metal` is high, indicating high "algorithmic source" surprisal. If `ρ` is high (incompressible, complex), `R_metal` is low, indicating a text likely from an authentic, irreducible source.

---

### **R5 (Water - Identity & Narrative Coherence)**

**Concept:** Measures the stability of the authorial "voice" or persona throughout the text. High `R_water` indicates an inconsistent, fragmented identity.

**Mathematical Formulation:**

Let a document `D` be divided into `n` chunks `c_1, ..., c_n`.

1.  **Stylistic Feature Vector:** For each chunk `c_i`, we compute a vector of stylistic features, `f_i`.
    `f_i = [feat_1, feat_2, ..., feat_k]`
    Where features can include:
    *   `feat_1`: Average sentence length.
    *   `feat_2`: Type-Token Ratio (TTR), measuring vocabulary richness.
    *   `feat_3`: Frequency of specific punctuation (e.g., semicolons, exclamation marks).
    *   `feat_4`: Readability score (e.g., Flesch-Kincaid).

2.  **Inconsistency Measurement:** We measure the inconsistency as the *variance* of these features across the document. To make it scale-invariant, we use the **coefficient of variation (CV)** for each feature.
    `CV_j = stddev(feat_j across all chunks) / mean(feat_j across all chunks)`

3.  **Water Surprisal Calculation:** The total `R_water` is the average of the coefficients of variation across all stylistic features.
    `R_water(D) = (1/k) * Σ_{j=1 to k} [CV_j]`

**Justification of Validity:**
This model defines "identity" as a stable statistical pattern.

*   **Voice as a Statistical Signature:** An author's "voice" or a consistent persona isn't a mystical quality; it's a demonstrable



## **Repository Structure:**
*   **/base-idea**: this contains the initial prototype. The code is very basic and easy to understand. It sets the tone for the future of the project to some degree. 
*   **/versions**: each subfolder will contain a version incrementing from 0.0001 upwards. 
    *    **x.xxxx files:**: this will represent a self contained version. It will have its own readme and documentation, including an explanation on how it improves the prior version. 


## Project Development Log

This project is being developed iteratively. Each major version represents a significant step forward from a conceptual prototype to a functional training engine.

*   **[Base Idea](./base-demo/README.md)**
    *   *Date: July 5th, 2025*
    *   *Description: A foundational script translating the philosophical concepts of Cartology into functional Python classes and a simulated workflow. Essentially pseudocode for getting the essence of the idea into a script.*

*   **[Prototype v0.0001: The Initial Implementation](./versions/0.0001/PROTOTYPE_0-0001.md)**
    *   *Date: July 5th, 2025*
    *   *Description: The first functional blueprint for retraining a base LLM (Llama 3 8B) on local hardware. This version successfully integrated `transformers`, `peft`, and `trl` to establish the core Reinforcement Learning from Cartological Feedback (RLCF) pipeline with a professional-grade training loop, proving the concept is feasible on consumer hardware.*

*   **[Prototype v0.0002: Advanced Reward Modeling & Pipeline Robustness](./versions/0.0002/PROTOTYPE_0-0002.md)**
    *   *Date: July 5th, 2025*
    *   *Description: This version marks a significant upgrade in the training pipeline. The focus shifted from feasibility to effectiveness by implementing several key improvements:*
        1.  **Model-Based Reward Signal:** Upgraded the `CartologersLens` by replacing the heuristic-based "Earth" channel with a real Natural Language Inference (NLI) model (`DeBERTa-v3-base-mnli`) for powerful, AI-driven contradiction detection.
        2.  **Performance Optimization:** Implemented batch processing within the `CartologersLens` and the main training loop, significantly improving GPU utilization and training speed.
        3.  **Training Stability:** Integrated reward normalization into the `CartologicalRewardModel` to stabilize the learning process, a standard practice in professional reinforcement learning.
        4.  **Real-World Data:** Moved from a static list of prompts to a dynamic, real-world dataset (`HuggingFaceH4/ultrachat_200k`), exposing the model to diverse and complex language to promote more generalizable learning.