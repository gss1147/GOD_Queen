import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from diffusers import StableDiffusionPipeline

from config import AI_MODELS_DIR, AI_CREATION_DIR

logger = logging.getLogger("HopeGeneration")
if not logger.handlers:
    (AI_CREATION_DIR / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(AI_CREATION_DIR / "logs" / "generation.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] generation: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


@dataclass
class GenerationPaths:
    code_dir: Path = AI_CREATION_DIR / "CODE"
    images_dir: Path = AI_CREATION_DIR / "Images"
    videos_dir: Path = AI_CREATION_DIR / "Video"
    music_dir: Path = AI_CREATION_DIR / "music"
    lyrics_dir: Path = AI_CREATION_DIR / "song lyrics"
    games_dir: Path = AI_CREATION_DIR / "Games"
    apps_dir: Path = AI_CREATION_DIR / "Apps"
    books_dir: Path = AI_CREATION_DIR / "books"
    animations_dir: Path = AI_CREATION_DIR / "animations"

    def ensure(self) -> None:
        for p in [
            self.code_dir,
            self.images_dir,
            self.videos_dir,
            self.music_dir,
            self.lyrics_dir,
            self.games_dir,
            self.apps_dir,
            self.books_dir,
            self.animations_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


class GenerationCore:
    """
    Core generative engine.

    - Uses DeepSeek Coder V2 Lite Instruct for all text/code.
    - Uses Stable Diffusion v1.5 for images (CPU).
    """

    def __init__(self) -> None:
        self.paths = GenerationPaths()
        self.paths.ensure()
        self._text_pipe: Optional[TextGenerationPipeline] = None
        self._sd_pipe: Optional[StableDiffusionPipeline] = None

    def _load_text_pipe(self) -> None:
        if self._text_pipe is not None:
            return
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
        self._text_pipe = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )

    def _load_sd_pipe(self) -> None:
        if self._sd_pipe is not None:
            return
        model_path = AI_MODELS_DIR / "stable-diffusion-stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        pipe = pipe.to("cpu")
        self._sd_pipe = pipe

    def generate_text_code(self, prompt: str, max_new_tokens: int = 512) -> Path:
        """
        Generate text/code with DeepSeek and save to a file.
        """
        self._load_text_pipe()
        sys_prompt = (
            "You are an advanced coding and content generator. "
            "Return only the final content, no commentary."
        )
        full_prompt = f"{sys_prompt}\n\nUser request:\n{prompt}\n\nOutput:"
        out = self._text_pipe(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self._text_pipe.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"]
        if text.startswith(full_prompt):
            text = text[len(full_prompt) :]
        text = text.strip()

        fname = f"hope_gen_{torch.randint(0, 1_000_000, ()).item()}.txt"
        path = self.paths.code_dir / fname
        path.write_text(text, encoding="utf-8")
        logger.info("Saved generated text/code to %s", path)
        return path

    def generate_image(self, prompt: str, steps: int = 25) -> Optional[Path]:
        """
        Generate a single image from a text prompt using Stable Diffusion.
        """
        try:
            self._load_sd_pipe()
        except Exception as exc:
            logger.error("Failed to load Stable Diffusion: %s", exc)
            return None

        image = self._sd_pipe(prompt, num_inference_steps=steps).images[0]
        fname = f"hope_image_{torch.randint(0, 1_000_000, ()).item()}.png"
        path = self.paths.images_dir / fname
        image.save(path)
        logger.info("Saved generated image to %s", path)
        return path