#run example: python generate_cg.py
import os
import subprocess

apk_dir = 'path/to/apks'      #CUSTOMIZE
output_dir = 'path/to/apks_cg'   #CUSTOMIZE 

os.makedirs(output_dir, exist_ok=True)

i = 0
skipped = 0

for root, _, files in os.walk(apk_dir):
    for file in files:
        if not file.endswith(".apk"):
            continue

        apk_path = os.path.join(root, file)
        output_file = os.path.join(output_dir, f"{file}_CG.gml")

        # Skip if call graph already exists
        if os.path.exists(output_file):
            skipped += 1
            print(f"SKIP ({skipped}): {file} – CG already exists")
            continue

        i += 1
        print(f"{i} - Processing {file}")

        try:
            subprocess.run(
                ["androguard", "cg", apk_path, "-o", output_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            print(f"Created CG for {file}")

        except subprocess.CalledProcessError as e:
            print(f"Failed to create CG for {file}: {e}")

print(f"Finished generating call graphs. Created: {i}, Skipped: {skipped}")
