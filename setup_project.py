import os

# The exact architecture we agreed upon
directories = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/explainability",
    "src/stability",
    "src/decision",
    "src/pipeline",
    "src/evaluation",
    "app",
    "reports/figures",
    "scripts"
]

files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "src/config.py",
    "src/data/loader.py",
    "src/data/splitter.py",
    "src/features/build_features.py",
    "src/models/garch.py",
    "src/models/xgboost_vol.py",
    "src/explainability/shap_explainer.py",
    "src/stability/shap_stability.py",
    "src/decision/confidence_gate.py",
    "scripts/run_pipeline.py"
]

def create_structure():
    print("Starting Project Setup...")
    
    # Create Directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   [DIR]  {directory}")

    # Create Files
    for file in files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                pass # Create empty file
            print(f"   [FILE] {file}")
        else:
            print(f"   [SKIP] {file} already exists")

    print("\nProject structure created successfully.")

if __name__ == "__main__":
    create_structure()