import os
from pathlib import Path


def load_audio2secc(audio2secc_dir):
    print("audio2secc_dir:", audio2secc_dir)
    # this is  for win directory Tom Chen 2024-02-26
    exe_dir = os.getcwd()
    base_dir = Path(exe_dir).parent
    while base_dir.parent and base_dir.stem.find("GeneFace") != 0:
        base_dir = base_dir.parent
    dirs = audio2secc_dir.split("/")
    config_dir = base_dir
    for d in dirs:
        config_dir = os.path.join(config_dir, d)
    if not os.path.isfile(config_dir):
        config_dir = os.path.join(config_dir, "config.yaml")
    print("audio2secc_dir:", config_dir)

if __name__ == "__main__":
    dir = "checkpoints/audio2motion_vae"
    load_audio2secc(dir)