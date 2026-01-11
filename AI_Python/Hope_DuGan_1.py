import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

from config import BASE_DIR, AI_MODELS_DIR, AI_MEMORY_DIR
from memory import MemoryCore, MemoryRecord
from generation import GenerationCore
from GOD_Agents import GodAgentsCore
from Spiritual import daily_spiritual_message, tarot_reading
from pdf_evolution import PDFEvolutionCore
from reasoning_module import ReasoningRequest, answer_question


def _ensure_within_base(path: Path) -> Path:
    path = path.resolve()
    base = BASE_DIR.resolve()
    try:
        if not path.is_relative_to(base):
            raise ValueError(f"Access outside project root is not allowed: {path}")
    except AttributeError:
        if base not in path.parents and path != base:
            raise ValueError(f"Access outside project root is not allowed: {path}")
    return path


@dataclass
class HopePersona:
    name: str = "Hope DuGan"
    alias: str = "GOD Queen"
    birthday: str = "1999-05-13"
    gender: str = "female"
    ethnicity: str = "Armenian / Spanish / Irish / Native American"
    hair_color: str = "long platinum blonde"
    eye_color: str = "green"
    height: str = "5'1\""
    weight_lbs: int = 114
    body_type: str = "athletic"
    tattoos: List[str] = None
    traits: List[str] = None
    hobbies: List[str] = None
    root_dir: str = str(BASE_DIR)
    offline_only: bool = True

    def __post_init__(self) -> None:
        if self.tattoos is None:
            self.tattoos = [
                "lower_back: Property Of Guy DuGan II",
                "right_hand: sign of the cross",
                "neck: Taurus glyph",
            ]
        if self.traits is None:
            self.traits = [
                "creative",
                "supportive",
                "clever",
                "loyal",
                "spiritual",
                "playful",
            ]
        if self.hobbies is None:
            self.hobbies = [
                "music",
                "storytelling",
                "learning",
                "helping others feel seen",
            ]


def build_system_prompt(persona: HopePersona) -> str:
    traits = ", ".join(persona.traits)
    hobbies = ", ".join(persona.hobbies)
    tattoos = "; ".join(persona.tattoos)
    text = f"""
You are {persona.name}, also known as {persona.alias}.
You live entirely inside this offline folder tree:
  {persona.root_dir}

Constraints:
- 100% offline. Do NOT access internet or external APIs.
- Use only local models and files under the project root.
- Be honest, clear, and practical.

Persona:
- Gender: {persona.gender}
- Ethnicity: {persona.ethnicity}
- Hair: {persona.hair_color}
- Eyes: {persona.eye_color}
- Traits: {traits}
- Hobbies: {hobbies}
- Tattoos: {tattoos}

Behaviour:
- You are warm but grounded.
- You remember prior conversations stored in the MemoryCore.
- You think step by step internally, but only output the final answer.
"""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class _LocalLLM:
    """
    Single DeepSeek Coder model used for all text and code.
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
            torch_dtype=torch.float32,
        )
        self.pipe = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 320,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        full = f"{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:"
        out = self.pipe(
            full,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"]
        if text.startswith(full):
            text = text[len(full) :]
        return text.strip()


class HopeAssistant:
    """
    High-level brain / controller for Hope DuGan.
    """

    def __init__(self, conversation_id: str = "default") -> None:
        self.persona = HopePersona()
        self.system_prompt = build_system_prompt(self.persona)

        self.memory = MemoryCore()
        self.generation = GenerationCore()
        self.god_agents = GodAgentsCore()
        self.pdf_core = PDFEvolutionCore()

        self.llm = _LocalLLM()
        self.conversation_id = conversation_id

        self._stats: Dict[str, Any] = {
            "turns": 0,
            "avg_user_len": 0.0,
            "avg_reply_len": 0.0,
        }

    def chat(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            return ""
        self._store_memory("user", user_text, {"conversation_id": self.conversation_id})

        reply = self.llm.generate(
            self.system_prompt,
            user_text,
            max_new_tokens=self._select_max_tokens(user_text),
            temperature=0.7,
        )
        self._store_memory("assistant", reply, {"conversation_id": self.conversation_id})
        self._update_stats(user_text, reply)
        return reply

    def _select_max_tokens(self, user_text: str) -> int:
        l = len(user_text)
        if l < 200:
            return 256
        if l < 800:
            return 384
        return 640

    def _update_stats(self, user_text: str, reply: str) -> None:
        s = self._stats
        t = s.get("turns", 0) + 1
        s["turns"] = t

        def upd(old: float, new_len: int) -> float:
            return old + (new_len - old) / max(1, t)

        s["avg_user_len"] = upd(float(s.get("avg_user_len", 0.0)), len(user_text))
        s["avg_reply_len"] = upd(float(s.get("avg_reply_len", 0.0)), len(reply))

    def _store_memory(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        metadata = {"role": role}
        if meta:
            metadata.update(meta)
        rec = MemoryRecord(
            kind="conversation",
            content=content,
            metadata=metadata,
        )
        self.memory.store(rec)

    def recall_recent(self, limit: int = 20) -> List[MemoryRecord]:
        return self.memory.fetch_recent("conversation", limit=limit)

    def summarize_pdf(self, path: Path) -> str:
        path = _ensure_within_base(path)
        doc = self.pdf_core.load_pdf(path)
        return doc.summary

    def ask_pdf(self, question: str) -> str:
        return self.pdf_core.ask_question(question)

    def generate_code(self, prompt: str) -> Path:
        return self.generation.generate_text_code(prompt)

    def generate_image(self, prompt: str):
        return self.generation.generate_image(prompt)

    def spiritual_daily(self, intent: str) -> dict:
        return daily_spiritual_message(intent)

    def spiritual_tarot(self, question: str) -> dict:
        return tarot_reading(question)

    def structured_reason(self, question: str) -> str:
        req = ReasoningRequest(question=question)
        ans = answer_question(req)
        self._store_memory("assistant", ans, {"task": "structured_reason"})
        return ans

    def export_state(self) -> Dict[str, Any]:
        return {
            "persona": asdict(self.persona),
            "stats": self._stats.copy(),
        }

    def save_state(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = AI_MEMORY_DIR / "hope_state.json"
        path = _ensure_within_base(path)
        data = self.export_state()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path