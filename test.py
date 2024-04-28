import pdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



# get the md text
md_text = pdf4llm.to_markdown("test2.pdf")  

# check for the md text
# output_file = "test.md"
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write(md_text)

# print(output_file is saved)

# split the md into chunks
splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)
splitter.create_documents([md_text])

# load embbeding model
# supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs)

vectordb = Chroma.from_documents(documents=splitter, embedding=embedding, persist_directory=persist_directory)

model_path = "llama.cpp/models/llama-2-7b-chat/llama-2_q4.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)


# DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""<<SYS>> 
#     You are a helpful assistant eager to assist with providing better Google search results.
#     <</SYS>> 
    
#     [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
#             relevant, and concise:
#             {question} 
#     [/INST]""",
# )

# DEFAULT_SEARCH_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are a helpful assistant eager to assist with providing better Google search results. \
#         Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
#         relevant, and concise: \
#         {question}""",
# )

# QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
#     default_prompt=DEFAULT_SEARCH_PROMPT,
#     conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
# )

# prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
# prompt


# llm_chain = LLMChain(prompt=prompt, llm=llm)
# question = "What is Taiwan known for?"
# llm_chain.invoke({"question": question})


# retriever = vectordb.as_retriever()

# qa = RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=retriever, 
#     verbose=True
# )

# query = "Tell me about Alison Hawk's career and age"
# qa.invoke(query)

