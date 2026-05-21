from src.data_loader import precompute_merge_captions

print("Starting caption merging...")
precompute_merge_captions(
    json_path="data/SECOND-CC-AUG.json",
    output_path="data/merged-SECOND-CC.json"
)
print("Merging finished.")