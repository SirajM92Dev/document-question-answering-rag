import argparse
from pathlib import Path

from .config import get_settings
from .pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs and ask document-grounded questions.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="Index a PDF")
    ingest.add_argument("path", type=Path)
    ask = subparsers.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--document-id")
    args = parser.parse_args()
    pipeline = RAGPipeline(get_settings())
    if args.command == "ingest":
        print(pipeline.ingest(args.path, args.path.name).model_dump_json(indent=2))
    else:
        print(pipeline.ask(args.question, args.document_id).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
