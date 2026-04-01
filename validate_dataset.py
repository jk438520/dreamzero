from datasets import load_dataset

# This forces the library to verify the checksums, splits, and number of examples
dataset = load_dataset(
    "data/droid_lerobot", 
    verification_mode="all_checks" 
)
print("Dataset loaded and verified successfully!")