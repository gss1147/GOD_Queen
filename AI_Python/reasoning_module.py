from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from config import AI_MODELS_DIR


@dataclass
class ReasoningRequest:
    """
    Request to the GOD Reasoning Core.

    Attributes:
        question:       Natural-language question or problem.
        mode:           One of:
                        - "hybrid"       (default: full hyper-hybrid reasoning stack)
                        - "symbolic"
                        - "probabilistic"
                        - "graph"
                        - "multimodal"
                        - "causal"
                        - "scientific"
        extra_context:  Optional extra facts, assumptions, or data.
    """
    question: str
    mode: str = "hybrid"
    extra_context: Optional[str] = None


class _ReasoningLLM:
    """
    Wrapper around the DeepSeek Coder V2 Lite Instruct model.

    This is the shared reasoning engine for:
    - CHAT CORE
    - GOD Agents Core
    - MEMORY CORE
    """

    def __init__(self) -> None:
        model_path = AI_MODELS_DIR / "deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # CPU-friendly on your system
        )

        self.pipe = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU only
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()


_reasoning_llm: Optional[_ReasoningLLM] = None


def _get_llm() -> _ReasoningLLM:
    global _reasoning_llm
    if _reasoning_llm is None:
        _reasoning_llm = _ReasoningLLM()
    return _reasoning_llm


# ---------------------------------------------------------------------------
# GOD Reasoning Core – description and modes
# ---------------------------------------------------------------------------

def describe_reasoning_capabilities() -> str:
    """
    Human-readable description of the GOD Reasoning Core and the AI
    technologies it conceptually combines.
    """
    return (
        "GOD Reasoning Core – Hyper-Hybrid AI Stack:\n"
        "\n"
        "1) Neural Networks & Transformers – Deep learning models inspired by the\n"
        "   brain, able to process large amounts of data, recognize patterns,\n"
        "   and perform sophisticated natural language reasoning.\n"
        "\n"
        "2) Probabilistic Models – Bayesian-style reasoning and Markov-style\n"
        "   thinking to handle uncertainty, evaluate likelihoods, and adapt\n"
        "   decisions based on incomplete information.\n"
        "\n"
        "3) Symbolic Reasoning – Logic- and rule-based reasoning that uses\n"
        "   explicit symbols, definitions, and inference rules, similar to how\n"
        "   mathematicians work through a proof step by step.\n"
        "\n"
        "4) Graph-Based Reasoning – Knowledge-graph style thinking in terms of\n"
        "   nodes (concepts) and edges (relations), tracing paths through\n"
        "   complex conceptual networks to uncover hidden connections.\n"
        "\n"
        "5) Multimodal AI (text-centric here) – Designed to conceptually fuse\n"
        "   text with other data types (vision, audio, numerical signals) for\n"
        "   richer reasoning when those modalities are available.\n"
        "\n"
        "6) Advanced Mathematics – Symbolic-style computation, equation\n"
        "   manipulation, approximate numeric reasoning, and optimization-style\n"
        "   thinking for physical and abstract problems.\n"
        "\n"
        "7) Science & Chemistry – Conceptual support for materials discovery,\n"
        "   drug design, and molecular reasoning: understanding structure,\n"
        "   reactivity, and mechanism at a high level of abstraction.\n"
        "\n"
        "8) Chirality & Quantum-style Reasoning – Awareness of symmetry,\n"
        "   chirality, spin-like distinctions, and subtle quantum-style\n"
        "   differences that can flip behaviour at small scales.\n"
        "\n"
        "9) Hyper-Hybrid Meta-Learning – Combining adaptive reasoning,\n"
        "   continual learning concepts, neuro-symbolic AI, probabilistic\n"
        "   reasoning, and higher-order abstraction to refine strategies over\n"
        "   time and solve complex, multi-step logical problems.\n"
        "\n"
        "10) Multimodal Fusion & Causal Inference – Fusing different forms of\n"
        "    information (text, imagined visuals, numbers) while aiming to\n"
        "    separate correlation from true cause–effect structure and to form\n"
        "    generative hypotheses that go beyond the obvious.\n"
    )


def _mode_instructions(mode: str) -> str:
    """
    Mode-specific steering text mapping the hyper-hybrid stack into
    concrete reasoning styles.
    """
    mode = (mode or "hybrid").lower()

    if mode == "symbolic":
        return (
            "Use symbolic and logic-based reasoning. Define symbols, state\n"
            "assumptions, and derive conclusions as if working through a formal\n"
            "proof or logic problem."
        )
    if mode == "probabilistic":
        return (
            "Use probabilistic reasoning. Make uncertainty explicit, discuss\n"
            "different scenarios and their likelihoods, and think in terms of\n"
            "conditional probabilities."
        )
    if mode == "graph":
        return (
            "Think in terms of a conceptual graph. Identify key entities and\n"
            "relations, then trace chains of connected ideas to explain how the\n"
            "answer emerges."
        )
    if mode == "multimodal":
        return (
            "Imagine you have access to diagrams, images, and numerical tables,\n"
            "but keep the answer in text. Describe what those hypothetical\n"
            "visuals would show and how they support your reasoning."
        )
    if mode == "causal":
        return (
            "Focus on cause and effect. Distinguish correlation from causation,\n"
            "describe mechanisms, and use counterfactuals (what if X changed?)\n"
            "to reason about the structure of the problem."
        )
    if mode == "scientific":
        return (
            "Answer like a scientific researcher. Use precise language, refer\n"
            "to physical or mathematical principles when helpful, and propose\n"
            "possible experiments or observations that could test your claims."
        )

    return (
        "Use a hyper-hybrid style that combines:\n"
        "- deep neural pattern recognition,\n"
        "- probabilistic reasoning,\n"
        "- symbolic and logical steps,\n"
        "- graph-style conceptual connections,\n"
        "- causal analysis,\n"
        "- and generative hypothesis formation.\n"
        "Reflect on your own reasoning and resolve contradictions before\n"
        "producing the final answer."
    )


def _build_system_prompt(req: ReasoningRequest) -> str:
    capabilities = describe_reasoning_capabilities()
    mode_text = _mode_instructions(req.mode)

    sys_prompt = f"""
You are the GOD Reasoning Core of the Hope DuGan AI system.

Your job:
- Understand complex questions deeply.
- Use the hyper-hybrid reasoning stack described below.
- Think step by step INTERNALLY.
- Output ONLY the final, cleaned-up answer (no raw scratch work).

Hyper-hybrid reasoning stack:
{capabilities}

Mode configuration:
{mode_text}

Constraints:
- You are 100% offline and cannot access the internet.
- Do NOT fabricate specific external data such as real-time statistics,
  web URLs, or unverifiable factual claims.
- You MAY propose hypotheses, theoretical mechanisms, or experiments,
  but clearly mark them as hypotheses or suggestions.
""".strip()

    return sys_prompt


def answer_question(req: ReasoningRequest) -> str:
    sys_prompt = _build_system_prompt(req)

    user_block = f"Question: {req.question.strip()}"
    if req.extra_context:
        user_block += (
            "\n\nExtra context (may or may not be relevant):\n"
            f"{req.extra_context.strip()}"
        )

    full = f"{sys_prompt}\n\n{user_block}\n\nAnswer:"
    return _get_llm().generate(full, max_new_tokens=640)


def quick_answer(question: str, mode: str = "hybrid") -> str:
    req = ReasoningRequest(question=question, mode=mode)
    return answer_question(req)


def structured_scientific_brief(question: str) -> str:
    req = ReasoningRequest(question=question, mode="scientific")
    return answer_question(req)