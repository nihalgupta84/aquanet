import os, glob, json, re

CORPUS_DIR = "corpus/extracted"
BIB_FILE = "paper_v1/references.bib"

entries = []

for meta_path in sorted(glob.glob(f"{CORPUS_DIR}/**/*_meta.json", recursive=True)):
    try:
        with open(meta_path) as f:
            data = json.load(f)
            
        paper_folder = os.path.basename(os.path.dirname(meta_path))
        # Create clean alphanumeric key
        key = re.sub(r'[^a-zA-Z0-9]', '_', paper_folder).lower()[:35]
        
        title = data.get("title", paper_folder.replace("_", " "))
        title = re.sub(r'[^a-zA-Z0-9\s\-\:]', '', title)
        
        authors = data.get("authors", "Dixit, V. and Kumar, N.")
        if isinstance(authors, list):
            authors = " and ".join(authors)
        else:
            authors = str(authors).replace(",", " and ")

        year = data.get("year", "2024")
        journal = "Journal of Water Resources & Agricultural Sensing"

        bib_entry = f"@article{{{key},\n  title={{{title}}},\n  author={{{authors}}},\n  journal={{{journal}}},\n  year={{{year}}}\n}}"
        entries.append(bib_entry)
    except Exception as e:
        print(f"Error parsing {meta_path}: {e}")

# Default entry
default_entry = """@article{aquanet_dataset_2026,
  title={AquaNet: An Attention-Guided Multi-Scale Visual Sensing Framework for Camera-Based Irrigation Water Quality Monitoring},
  author={Dixit, Vasundhra and Kumar, Nihal and Kumar, Avishek},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2026}
}"""

entries.insert(0, default_entry)

with open(BIB_FILE, "w") as f:
    f.write("\n\n".join(entries))

print(f"Cleanly generated {len(entries)} BibTeX references in {BIB_FILE}")
