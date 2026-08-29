import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


scripts = [
    "prepare_data.py",
    "add_history_features.py",
    "compare_models.py",
    "train_model.py",
]


for script in scripts:
    print()
    print("Running", script)

    subprocess.check_call([
        sys.executable,
        str(ROOT / "src" / script),
    ])


print()
print("Finished.")
print("Run the app with:")
print("streamlit run src/app.py")
