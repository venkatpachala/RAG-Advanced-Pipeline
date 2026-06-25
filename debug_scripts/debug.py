import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="json",
    save_images=True,
    specialized_image_parsing=True,
)

result = parser.get_json_result("test_data/research_papers/sample.pdf")

download_dir = "debug_images"

os.makedirs(download_dir, exist_ok=True)

parser.get_images(result, download_dir)

print(os.listdir(download_dir))