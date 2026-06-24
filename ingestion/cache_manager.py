import hashlib
import json
from pathlib import Path
from typing import List, Optional

from models.document_element import DocumentElement


class IngestionCache:
    def __init__(self, cache_dir: str = "cache/ingestion"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, file_path: str, parser_version: str) -> Path:
        source = Path(file_path)
        stat = source.stat()
        cache_key = hashlib.sha256(
            f"{source.resolve()}::{stat.st_mtime_ns}::{stat.st_size}::{parser_version}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.json"

    def get(self, file_path: str, parser_version: str) -> Optional[List[DocumentElement]]:
        path = self._cache_path(file_path, parser_version)
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return [DocumentElement.model_validate(item) for item in data]

    def set(self, file_path: str, parser_version: str, elements: List[DocumentElement]) -> None:
        path = self._cache_path(file_path, parser_version)
        data = [element.model_dump(mode="json") for element in elements]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
