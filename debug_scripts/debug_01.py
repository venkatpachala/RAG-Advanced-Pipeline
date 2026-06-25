import os
import json
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="json",
    save_images=True,
    specialized_image_parsing=True,
    extract_layout=True,
)

result = parser.get_json_result(
    "test_data/research_papers/sample.pdf"
)

pages = result[0]["pages"]

print("=" * 100)

for page in pages:

    print(f"\nPAGE {page['page']}")

    items = page.get("items", [])

    found = False

    for idx, item in enumerate(items):

        item_type = item.get("type", "").lower()

        if item_type in [
            "figure",
            "image",
            "diagram",
            "chart",
            "picture"
        ]:

            found = True

            print("\n" + "=" * 80)
            print(f"ITEM #{idx}")
            print(f"TYPE : {item_type}")

            print("\nRAW JSON")
            print(json.dumps(item, indent=2))

            print("\nBBox")
            print(item.get("bBox"))

    if not found:
        print("No figure/image items on this page.")