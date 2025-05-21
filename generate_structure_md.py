import os

# Names to ignore in the tree (hidden/system/cache)
EXCLUDE = {".git", ".venv", "__pycache__", ".DS_Store", ".ipynb_checkpoints"}

def generate_tree(path=".", indent="", is_last=True):
    tree_str = ""
    items = sorted(
        [item for item in os.listdir(path) if item not in EXCLUDE and not item.startswith(".")]
    )

    for index, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_last_item = index == len(items) - 1
        branch = "└── " if is_last_item else "├── "
        tree_str += f"{indent}{branch}{item}\n"

        if os.path.isdir(item_path):
            extension = "    " if is_last_item else "│   "
            tree_str += generate_tree(item_path, indent + extension, is_last_item)

    return tree_str

def save_structure_to_md(output_file="PROJECT_STRUCTURE.md"):
    header = "# 📁 Project Structure\n\n```\n"
    footer = "```"
    tree_content = generate_tree(".")
    full_content = header + tree_content + footer

    with open(output_file, "w") as f:
        f.write(full_content)

    print(f"✅ Project structure saved to {output_file}")

if __name__ == "__main__":
    save_structure_to_md()
