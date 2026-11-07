import argparse
import logging
import sys
from pathlib import Path

import pypdf
from google.genai.errors import APIError

from src.rag.chunking import chunk_documents
from src.rag.embeddings import GeminiEmbedder
from src.rag.vector_store import FaissVectorStore


logger = logging.getLogger(__name__)


def _read_document(path: Path) -> dict[str, object]:
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(path)
        content = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif path.suffix.lower() == ".txt":
        content = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported document type: {path.name}")
    return {"name": path.name, "content": content}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or search the Gemini + FAISS source-document index."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="PDF or TXT files to chunk and index",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Search the saved index instead of adding documents",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("vector_db"),
        help="Directory for the persisted FAISS index (default: vector_db)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.search is not None and args.files:
        parser.error("provide either --search or document files, not both")
    if args.search is None and not args.files:
        parser.error("provide at least one PDF/TXT file or use --search")

    try:
        store = FaissVectorStore(args.index_dir)
        if args.search is not None:
            results = store.search(args.search)
            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. cosine={result['score']:.4f} "
                    f"metadata={result['metadata']}\n{result['text']}\n"
                )
            return 0

        documents = [_read_document(path) for path in args.files]
        chunks = chunk_documents(documents)
        added = store.add_chunks(chunks, GeminiEmbedder()) if chunks else 0
        print(
            f"Embedded and stored {added} new chunk(s); "
            f"FAISS now contains {store.count} vector(s)."
        )
        return 0
    except (
        APIError,
        OSError,
        ValueError,
        RuntimeError,
        pypdf.errors.PdfReadError,
    ) as error:
        logger.exception("Could not complete the vector-store operation")
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
