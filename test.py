from pathlib import Path

ROOT = Path(r"D:\MELD_Project\repo_style_final")

models = {
    "Cross-modal attention":
        ROOT / "results/github_clean_new/checkpoint.pt",

    "Reliability-aware fusion":
        ROOT / "results/reliability_aware_new/reliability_checkpoint_new.pt",

    "Temporal GRU":
        ROOT / "results/temporal_svm_tuned_github/temporal_checkpoint.pt",

    "RBF-SVM":
        ROOT / "results/temporal_class_corrected/svm_temporal.joblib",

    "Reliability embeddings":
        ROOT / "results/reliability_aware_balanced_fix/reliability_embeddings.npz",

    "Temporal embeddings":
        ROOT / "results/temporal_svm_tuned_github/temporal_embeddings.npz",
}

print("\n===== SAVED MODEL VERIFICATION =====")

all_found = True

for name, path in models.items():
    status = path.is_file()
    print(f"{name}: {'FOUND' if status else 'MISSING'}")
    print(path)

    if not status:
        all_found = False

print("\nAll required files found:", all_found)