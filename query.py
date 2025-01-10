import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from embeddings import get_ollama_embedding_function
from langchain_ollama import OllamaLLM

CHROMA_PATH = './chroma'
PROMPT_TEMPLATE = """
This is the resume for Anuj Verma. Here are resume details
{context}

---
Answer the question based on the above details if matches: {question}
"""


def main():
  parser = argparse.ArgumentParser(description='Enter a query text to search in pdfs.')
  parser.add_argument('query_text', type=str, help='Query text')
  args = parser.parse_args()
  query_text = args.query_text
  query(query_text)
  

def query(query_text: str):
  chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_ollama_embedding_function())
  results = chroma.similarity_search_with_score(query_text, k=5)
  context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  ollama = OllamaLLM(model="phi4")
  response_text = ollama.invoke(prompt)
  
  print(response_text)

if __name__ == "__main__":
    main()