import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all Text files"""
    print(f"Loading text files from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your company files."
        )

    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. Please add your files."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}: ")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def main():
    print("Main fucntion.")
    # 1. Load files
    load_documents()
    # 2. Chunking the files
    # 3. Embedding and storing in Vector DB


if __name__ == "__main__":
    main()
