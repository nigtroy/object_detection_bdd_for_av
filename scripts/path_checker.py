import os

# 1. DEFINE YOUR PATHS EXACTLY AS YOU DID IN YOUR CODE
img_rel_path = r"C:\Users\gsamu\object_Detection_waymo\dataset\val_images"
lbl_rel_path = r"C:\Users\gsamu\object_Detection_waymo\annotations\val"

# 2. PRINT CURRENT WORKING DIRECTORY
print(f"Current Working Directory: {os.getcwd()}")
print("-" * 30)

# 3. CHECK IMAGE PATH
img_abs_path = os.path.abspath(img_rel_path)
print(f"Checking Image Path: {img_abs_path}")

if not os.path.exists(img_abs_path):
    print("❌ ERROR: Image folder does NOT exist here.")
else:
    files = os.listdir(img_abs_path)
    jpg_count = len([f for f in files if f.endswith('.jpg')])
    print(f"✅ FOUND folder. It contains {len(files)} files.")
    print(f"   --> {jpg_count} are .jpg images.")

print("-" * 30)

# 4. CHECK LABEL PATH
lbl_abs_path = os.path.abspath(lbl_rel_path)
print(f"Checking Label Path: {lbl_abs_path}")

if not os.path.exists(lbl_abs_path):
    print("❌ ERROR: Label folder does NOT exist here.")
else:
    files = os.listdir(lbl_abs_path)
    json_count = len([f for f in files if f.endswith('.json')])
    print(f"✅ FOUND folder. It contains {len(files)} files.")
    print(f"   --> {json_count} are .json files.")