from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from embeddings import get_ollama_embedding_function


DATA_PATH = './data'
CHROMA_PATH = './chroma'

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)
    


def add_to_chroma(chunks: list[Document]):
  chroma = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_ollama_embedding_function()
  )

  chunks_with_ids = calculate_chunk_ids(chunks)
  existing_items = chroma.get(include=[]) 
  existing_ids = set(existing_items["ids"])
  
  new_chunks = []
  for chunk in chunks_with_ids:
    if chunk.metadata["id"] not in existing_ids:
      new_chunks.append(chunk)
  
  if len(new_chunks):
    print(f"Adding {len(new_chunks)} new chunks to Database.")
    new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
    chroma.add_documents(new_chunks, id=new_chunks_ids)
  else:
    print("No new chunks to add to Database.")


def calculate_chunk_ids(chunks):
  last_page_id = None
  current_chunk_index = 0

  for chunk in chunks:
      source = chunk.metadata.get("source")
      page = chunk.metadata.get("page")
      current_page_id = f"{source}:{page}"

      # If the page ID is the same as the last one, increment the index.
      if current_page_id == last_page_id:
          current_chunk_index += 1
      else:
          current_chunk_index = 0

      # Calculate the chunk ID.
      chunk_id = f"{current_page_id}:{current_chunk_index}"
      last_page_id = current_page_id

      # Add it to the page meta-data.
      chunk.metadata["id"] = chunk_id

  return chunks

documents = load_documents()

if(documents):
  chunks = split_documents(documents)
  add_to_chroma(chunks)