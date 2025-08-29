import os
import subprocess
import sys

BASE_DIR = "classroom_engagement"
PYPROJECT_PATH = os.path.join(BASE_DIR, "pyproject.toml")

def create_venv():
    venv_path = ".venv"
    if not os.path.isdir(venv_path):
        print("🐍 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    else:
        print("⚠️ Virtual environment already exists.")

def install_deps():
    pip_path = os.path.join(".venv", "bin", "pip")
    if not os.path.exists(pip_path):
        print("❌ pip not found in virtual environment. Please check your Python setup.")
        return

    print("📦 Installing dependencies from pyproject.toml...")
    subprocess.run([pip_path, "install", "-e", BASE_DIR], check=True)

if __name__ == "__main__":
    print("🚀 Setting up the virtual environment and installing dependencies...\n")
    create_venv()
    install_deps()
    print("\n✅ Done! To activate:")
    print("   source .venv/bin/activate")
    print("Then run your demo with:")
    print(f"   python {BASE_DIR}/scripts/run_emotion_demo.py")