import os
import json
import asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="json",
    save_images=True,
    specialized_image_parsing=True,
)

async def main():

    result = await parser.aget_json_result(
        "test_data/research_papers/sample.pdf"
    )

    page = result[0]["pages"][0]

    print("=" * 80)
    print("PAGE KEYS")
    print(page.keys())

    print("=" * 80)
    print("IMAGES")
    print(json.dumps(page.get("images", []), indent=2))

    print("=" * 80)
    print("ITEM TYPES")

    for item in page["items"]:
        print(item.get("type"))

asyncio.run(main())