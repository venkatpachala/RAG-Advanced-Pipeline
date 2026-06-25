from pathlib import Path

print("=" * 60)

download_root = Path("extracted_images")

print("Root exists:", download_root.exists())

if download_root.exists():

    for folder in download_root.iterdir():

        print(f"\nFolder: {folder}")

        files = list(folder.glob("*"))

        print("Number of files:", len(files))

        for f in files:
            print("  ", f.name)

else:
    print("No extracted_images directory found.")