from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 1. Inicializar Gemini y Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Cargar y procesar el PDF
loader = PyMuPDFLoader("TP Final Consolidación-1.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# =====================================================================
# 3. Guardar en la Vector Store y crear el Retriever (el buscador)
# =====================================================================
# 1. Inicializamos Chroma pasándole la función de embedding
vector_store = Chroma(embedding_function=embeddings)

# 2. Agregamos los chunks de forma explícita
vector_store.add_documents(documents=chunks)

# 3. Creamos el retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Definir el comportamiento del LLM (El Prompt)
system_prompt = (
    "Sos un asistente experto en analizar documentos.\n"
    "Respondé la pregunta del usuario basándote ÚNICAMENTE en el siguiente contexto extraído del PDF:\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Armar la cadena de ejecución lineal (Sin agentes, sin grafos)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

if __name__ == "__main__":
    print("\n🚀 Sistema RAG Iniciado. Escribí tu pregunta sobre el PDF (o 'salir' para terminar):\n")
    
    while True:
        user_input = input("👤 Yo: ")
        
        # Condición de salida
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("¡Nos vemos!")
            break
            
        if not user_input.strip():
            continue
            
        # Invocamos la cadena con la pregunta del usuario
        response = rag_chain.invoke({"input": user_input})
        
        print("\n--- 🤖 Respuesta del PDF ---")
        print(response["answer"])
        print("-" * 30 + "\n")