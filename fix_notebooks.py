import json
from pathlib import Path

# python fix_notebooks.py
# This script fixes the notebooks by removing the broken widget metadata and the widget outputs from the cells.

for nb_path in sorted(Path("notebooks").glob("*.ipynb")):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False

    # 1. Remove broken notebook-level widget metadata
    if nb.get("metadata", {}).pop("widgets", None) is not None:
        changed = True

    # 2. Remove widget outputs from cells (keep text/plain fallbacks)
    for cell in nb.get("cells", []):
        new_outputs = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "display_data":
                data = output.get("data", {})
                if "application/vnd.jupyter.widget-view+json" in data:
                    changed = True
                    data = {
                        k: v for k, v in data.items()
                        if k != "application/vnd.jupyter.widget-view+json"
                    }
                    if not data:
                        continue
                    output = {**output, "data": data}
            new_outputs.append(output)
        cell["outputs"] = new_outputs

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"fixed: {nb_path}")
    else:
        print(f"ok:    {nb_path}")