## 📝 Descripción del Proyecto

**pdf-reader-rag** es una aplicación de consola en Python que implementa una arquitectura **RAG (Retrieval-Augmented Generation)** lineal y eficiente para interactuar con documentos PDF de forma local y dinámica. 

El sistema permite cargar documentos extensos (como auditorías, manuales o informes técnicos), fragmentarlos de manera inteligente y realizar consultas en lenguaje natural, garantizando respuestas precisas basadas **únicamente** en el contexto del documento provisto.

### ⚙️ Arquitectura y Flujo de Datos

A diferencia de los enfoques basados en agentes autónomos, este proyecto implementa una **cadena RAG lineal (Stuff Documents Chain)** optimizada para garantizar predictibilidad, velocidad y un consumo mínimo de recursos:

1. **Ingesta y Segmentación:** Se procesa el PDF utilizando `PyMuPDFLoader` y se fragmenta el texto en chunks mediante un divisor de caracteres recursivo (`RecursiveCharacterTextSplitter`), manteniendo un solapamiento para no perder el contexto semántico entre fragmentos.
2. **Embeddings Locales:** Para evitar las limitaciones de cuota (Rate Limits / Error 429) de las APIs en la nube, la vectorización se realiza de manera 100% local utilizando el modelo open-source `all-MiniLM-L6-v2` a través de **HuggingFace**.
3. **Almacenamiento Vectorial:** Los vectores resultantes se indexan en una base de datos embebida **Chroma**, permitiendo búsquedas de similitud ultrarrápidas.
4. **Recuperación y Síntesis (LLM):** Al realizar una pregunta, el componente *Retriever* aísla los 3 fragmentos más relevantes del documento y los inyecta en un prompt estructurado. Finalmente, el modelo **Gemini 2.5 Flash** procesa este contexto refinado para generar una respuesta natural y contextualizada.

### 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.14
* **Orquestación de IA:** LangChain (`langchain-core`, `langchain-chroma`)
* **Modelo de Lenguaje (LLM):** Google Gemini 2.5 Flash (vía `langchain-google-genai`)
* **Generación de Embeddings:** HuggingFace / Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Store:** Chroma DB
* **Procesamiento de PDF:** PyMuPDF (`fitz`)