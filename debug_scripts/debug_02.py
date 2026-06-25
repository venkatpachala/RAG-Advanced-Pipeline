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
)

result = parser.get_json_result(
    "test_data/research_papers/sample.pdf"
)

print("=" * 80)

for page in result[0]["pages"]:

    print(f"\nPAGE {page['page']}")

    print("\nImages")

    print(json.dumps(page.get("images", []), indent=2))

    print("\nCharts")

    print(json.dumps(page.get("charts", []), indent=2))