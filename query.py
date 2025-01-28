def warn(*args, **kwargs):

     pass

import warnings

warnings.warn = warn

warnings.filterwarnings("ignore")

from langchain.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

import wget

from google.colab import files



from langchain.vectorstores import FAISS

from langchain.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


uploaded = files.upload()


for file_name in uploaded.keys():
    document_path = file_name
    print(f"Uploaded file: {document_path}")


loader = TextLoader(document_path, encoding="ISO-8859-1")
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


print(f"Number of chunks: {len(docs)}")

parameters = {
    GenParams.MIN_NEW_TOKENS: 130,
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5

}

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    **parameters
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define embedding model for document retrieval
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embedding_model)

# Save the index to the desired path
db.save_local("C:/Users/donpd/project2/faiss_index")

docsearch = FAISS.load_local(
    "C:/Users/donpd/project2/faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Set up QA system
retriever = docsearch.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Query
query = "what are the legal points"
result = qa.run(query)
print(result)
