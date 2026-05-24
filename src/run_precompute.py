from data_loader import precompute_merge_captions

print("Starting caption merging...")
precompute_merge_captions(
    json_path="/data2/data/SECOND-CC-AUG.json",
    output_path="/data2/merged-SECOND-CC.json"
)
print("Merging finished.")