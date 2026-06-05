from data_loader import precompute_merge_captions

"""
Generates 1 single caption from the original 5 captions of the Second-CC dataset.
"""

print("Starting caption merging...")
precompute_merge_captions(
    json_path="/data2/data/SECOND-CC-AUG.json",
    output_path="/data2/merged-SECOND-CC.json"
)
print("Merging finished.")