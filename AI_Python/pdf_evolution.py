import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

from config import AI_MODELS_DIR, AI_MEMORY_DIR
from memory import MemoryCore, MemoryRecord

logger = logging.getLogger("HopePDF")
if not logger.handlers:
    (AI_MEMORY_DIR / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(AI_MEMORY_DIR / "logs" / "pdf_evolution.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] pdf: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


@dataclass
class KnowledgeDoc:
    path: Path
    summary: str
    pages: int


class _PDFLLM:
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


_pdf_llm: _PDFLLM | None = None


def _get_pdf_llm() -> _PDFLLM:
    global _pdf_llm
    if _pdf_llm is None:
        _pdf_llm = _PDFLLM()
    return _pdf_llm


class PDFEvolutionCore:
    """
    Lightweight PDF ingestion and Q&A core.
    """

    def __init__(self) -> None:
        self.memory = MemoryCore()
        self.current_doc: KnowledgeDoc | None = None
        self.current_text: str = ""

    def load_pdf(self, path: Path) -> KnowledgeDoc:
        path = Path(path)
        text_chunks: List[str] = []
        pages = 0
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages += 1
                txt = page.extract_text() or ""
                if txt.strip():
                    text_chunks.append(txt.strip())
        raw_text = "\n\n".join(text_chunks)
        self.current_text = raw_text

        llm = _get_pdf_llm()
        prompt = (
            "Summarize the following document in 8–12 bullet points. "
            "Focus on the most important facts and ideas.\n\n"
            f"=== DOCUMENT TEXT START ===\n{raw_text[:12000]}\n=== DOCUMENT TEXT END ===\n\nSummary:"
        )
        summary = llm.generate(prompt, max_new_tokens=512)

        doc = KnowledgeDoc(path=path, summary=summary, pages=pages)
        self.current_doc = doc

        rec = MemoryRecord(
            kind="pdf_summary",
            content=summary,
            metadata={"path": str(path), "pages": pages},
        )
        self.memory.store(rec)
        logger.info("Loaded PDF %s (%s pages)", path, pages)
        return doc

    def ask_question(self, question: str) -> str:
        if not self.current_text:
            return "No PDF is currently loaded."
        llm = _get_pdf_llm()
        prompt = (
            "You are an expert reader of the following document. "
            "Answer the user's question concisely, using only information "
            "that can be inferred from the text.\n\n"
            f"=== DOCUMENT TEXT START ===\n{self.current_text[:16000]}\n=== DOCUMENT TEXT END ===\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        answer = llm.generate(prompt, max_new_tokens=512)
        rec = MemoryRecord(
            kind="pdf_qa",
            content=answer,
            metadata={"question": question},
        )
        self.memory.store(rec)
        return answer