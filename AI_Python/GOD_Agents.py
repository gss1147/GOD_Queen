import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any, List

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

from config import AI_MODELS_DIR, AI_CREATION_DIR

logger = logging.getLogger("HopeGODAgents")
if not logger.handlers:
    (AI_CREATION_DIR / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(AI_CREATION_DIR / "logs" / "god_agents.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] god_agents: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


@dataclass
class GodAgentTask:
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict


class _GodAgentsLLM:
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

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()


_god_llm: _GodAgentsLLM | None = None


def _get_god_llm() -> _GodAgentsLLM:
    global _god_llm
    if _god_llm is None:
        _god_llm = _GodAgentsLLM()
    return _god_llm


class GodAgentsCore:
    """
    Background auto-agent manager.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[GodAgentTask]" = queue.Queue()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        self.army_active = False
        self.scout_active = False
        self.reporter_active = False
        self.code_wizards_active = False
        self.self_rewrite_active = False

    def submit_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._queue.put(GodAgentTask(fn=fn, args=args, kwargs=kwargs))

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                task.fn(*task.args, **task.kwargs)
            except Exception as exc:
                logger.error("GOD Agents task error: %s", exc)
            finally:
                self._queue.task_done()

    def shutdown(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)

    def enable_god_queen(self) -> None:
        logger.info("GOD QUEEN enabled (logical flag only).")

    def run_army_defense(self) -> None:
        self.army_active = True
        try:
            prompt = (
                "You are a security-focused code analyst. "
                "List 5 concrete checks this project should perform to stay safe, "
                "offline, and within its folder tree."
            )
            txt = _get_god_llm().generate(prompt, max_new_tokens=256)
            log_path = AI_CREATION_DIR / "CODE" / "god_army_defense.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(txt, encoding="utf-8")
            logger.info("GOD Agents army defense notes written to %s", log_path)
        finally:
            self.army_active = False

    def run_scout(self, urls: List[str]) -> None:
        self.scout_active = True
        try:
            txt = "Scout requested on URLs (offline log only):\n" + "\n".join(urls)
            log_path = AI_CREATION_DIR / "CODE" / "god_scout_log.txt"
            log_path.write_text(txt, encoding="utf-8")
            logger.info("GOD Agents scout log written to %s", log_path)
        finally:
            self.scout_active = False

    def run_reporter(self) -> None:
        self.reporter_active = True
        try:
            prompt = (
                "Write a short status report as Hope DuGan describing what the project "
                "is and what it is trying to become."
            )
            txt = _get_god_llm().generate(prompt, max_new_tokens=256)
            log_path = AI_CREATION_DIR / "CODE" / "god_reporter_status.txt"
            log_path.write_text(txt, encoding="utf-8")
            logger.info("GOD Agents reporter report written to %s", log_path)
        finally:
            self.reporter_active = False

    def run_code_wizards(self) -> None:
        self.code_wizards_active = True
        try:
            prompt = (
                "You are a senior code wizard. List 10 concrete, safe refactorings that "
                "could improve the Hope DuGan Python project without changing behaviour."
            )
            txt = _get_god_llm().generate(prompt, max_new_tokens=256)
            log_path = AI_CREATION_DIR / "CODE" / "god_code_wizards_ideas.txt"
            log_path.write_text(txt, encoding="utf-8")
            logger.info("GOD Agents code wizard notes written to %s", log_path)
        finally:
            self.code_wizards_active = False

    def run_self_rewrite(self) -> None:
        self.self_rewrite_active = True
        try:
            prompt = (
                "Imagine you are an AI maintaining your own source code. "
                "Write a checklist of steps to safely propose self-improvements, "
                "including human review checkpoints."
            )
            txt = _get_god_llm().generate(prompt, max_new_tokens=256)
            log_path = AI_CREATION_DIR / "CODE" / "god_self_rewrite_plan.txt"
            log_path.write_text(txt, encoding="utf-8")
            logger.info("GOD Agents self-rewrite plan written to %s", log_path)
        finally:
            self.self_rewrite_active = False

    def manual_search(self, query: str) -> None:
        prompt = (
            f"You are a research assistant with no internet access. Based only on "
            f"your training, provide a compact briefing on: {query}"
        )
        txt = _get_god_llm().generate(prompt, max_new_tokens=256)
        log_path = AI_CREATION_DIR / "CODE" / "god_manual_search.txt"
        log_path.write_text(txt, encoding="utf-8")
        logger.info("GOD Agents manual search results written to %s", log_path)