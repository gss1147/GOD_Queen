"""
Configuration module for Hope DuGan AI System.
Defines all important paths for the system.
"""
from pathlib import Path

# Base directory of the entire Hope DuGan project
BASE_DIR = Path(__file__).resolve().parents[1]  # Go up two levels from AI_Python to Hope_DuGan

# Define all the required directories according to the project structure
AI_MEMORY_DIR = BASE_DIR / "AI_Memory"
AI_PYTHON_DIR = BASE_DIR / "AI_Python"
AI_MODELS_DIR = BASE_DIR / "AI_Models"
AI_CREATION_DIR = BASE_DIR / "AI_Creation"
AI_MEDIA_DIR = BASE_DIR / "AI_Media"
AI_CONFIG_DIR = BASE_DIR / "AI_Config"

# Create directories if they don't exist
for dir_path in [AI_MEMORY_DIR, AI_PYTHON_DIR, AI_MODELS_DIR, AI_CREATION_DIR, AI_MEDIA_DIR, AI_CONFIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Subdirectories that need to be created
MEMORY_SUBDIRS = [
    "chroma_db",
    "GOD_Agents_db", 
    "logs",
    "memory",
    "Memory_Graphs",
    "personaL_DIARY",
    "personal_goals",
    "spiritual_guidance_db"
]

for subdir in MEMORY_SUBDIRS:
    (AI_MEMORY_DIR / subdir).mkdir(parents=True, exist_ok=True)

CREATION_SUBDIRS = [
    "animations",
    "Apps",
    "books",
    "CODE",
    "Games",
    "Images",
    "music",
    "song lyrics",
    "Video"
]

for subdir in CREATION_SUBDIRS:
    (AI_CREATION_DIR / subdir).mkdir(parents=True, exist_ok=True)

MEDIA_SUBDIRS = [
    "audio",
    "documents",
    "images",
    "json_code",
    "PDF Files",
    "python_code",
    "video"
]

for subdir in MEDIA_SUBDIRS:
    (AI_MEDIA_DIR / subdir).mkdir(parents=True, exist_ok=True)

__all__ = [
    "BASE_DIR",
    "AI_MEMORY_DIR",
    "AI_PYTHON_DIR",
    "AI_MODELS_DIR",
    "AI_CREATION_DIR",
    "AI_MEDIA_DIR",
    "AI_CONFIG_DIR",
]