import os

def print_tree_to_file(startpath, file, indent=''):
    for item in os.listdir(startpath):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            file.write(f"{indent}📁 {item}\n")
            print_tree_to_file(path, file, indent + "    ")
        else:
            file.write(f"{indent}📄 {item}\n")

# Usage
root_dir = r'C:\Users\Ahmed\OneDrive\Bureau\Study\PFA\FresNet_Change_Detection_Model\data'
with open("folder_tree.txt", "w", encoding='utf-8') as f:
    print_tree_to_file(root_dir, f)

print("Tree saved to folder_tree.txt")
