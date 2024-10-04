from typing import Dict, List, TypedDict
from fastapi import FastAPI, HTTPException
import time
from threading import Lock
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.middleware.cors import CORSMiddleware

from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import FAISS
import json
from langchain_community.tools import DuckDuckGoSearchResults
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document

### LLM
from langchain_ollama import ChatOllama
from pydantic import BaseModel

local_llm = 'SpeakLeash/bielik-11b-v2.2-instruct-imatrix:Q8_0'
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format='json')

import operator
from typing_extensions import TypedDict
from typing import List, Annotated

##########################################################################
# Kod Bartka      Kod Bartka    Kod Bartka     Kod Bartka     Kod Bartka #
##########################################################################

# Prompt templates
from prompts import router_instructions2, doc_grader_instructions2, doc_grader_prompt2, rag_prompt2, answer_grader_instructions2, answer_grader_prompt2, hallucination_grader_instructions2, hallucination_grader_prompt2


#LLM
local_llm2 = 'SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M'
llm2 = ChatOllama(model=local_llm2, temperature=0.2)
llm_json_mode2 = ChatOllama(model=local_llm2, temperature=0.2, format='json')

#Embedding model
embeddings2 = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

#Loading vectorstore
vectorstore2 = FAISS.load_local("my_faiss_store", embeddings=embeddings2, allow_dangerous_deserialization=True)
retriever2 = vectorstore2.as_retriever(k=3)

#Post-processing
def format_docs(docs2):
    return "\n\n".join(doc2.page_content for doc2 in docs2)

#Web search
web_search_tool2 = DuckDuckGoSearchResults(k=3)


#GraphState class
class GraphState2(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    history : str # LLM prompt
    generation : str # LLM generation
    web_search : str # Binary decision to run web search
    max_retries : int # Max number of retries for answer generation 
    answers : int # Number of answers generated
    loop_step: Annotated[int, operator.add] 
    documents : List[str] # List of retrieved documents


#Graph nodes (actions functions)
def rerank(query):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    embeddings2 = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

    vectorstore2 = FAISS.load_local("my_faiss_store", embeddings=embeddings2, allow_dangerous_deserialization=True)

    results = vectorstore2.similarity_search(query, k=50)

    chunks = [result.page_content for result in results]

    print("Started reranking...")

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank_chunks(query, chunks):
        inputs = [query + " [SEP] " + chunk for chunk in chunks]
        tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            scores = outputs.logits.squeeze().cpu().tolist() 

        reranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
        return reranked_chunks

    reranked_chunks = rerank_chunks(query, chunks)

    for i, chunk in enumerate(reranked_chunks[:10]):
        print(f"Rank {i+1}: {chunk}")

    return reranked_chunks[:6]
def retrieve2(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    documents = rerank(state["question"])
    return {"documents": documents}

def generate2(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    history = state["history"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    
    # RAG generation
    docs_txt2 = format_docs(documents)
    json_format = "Ważne byś wszystkie swoje opowiedzie przedstawiał w postaci zwykłego tekstu pisanego (zgodnie z promptami), czyli nie zwracał nic w stylu 'Question':some random text, 'Answer':some other random text. Just give me the text,"
    rag_prompt_formatted2 = rag_prompt2.format(context=docs_txt2, history=history, question=question)
    generation2 = llm2.invoke([SystemMessage(content=json_format), HumanMessage(content=rag_prompt_formatted2)])
    return {"generation": generation2, "loop_step": loop_step+1}

def grade_documents2(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    
    documents = state["documents"]
    history = state["history"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No" 

    relevant = 0
    irreleavnt = 0

    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt2.format(document=d, history=history, question=question)
        result = llm_json_mode2.invoke([SystemMessage(content=doc_grader_instructions2)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']
        # Document relevant
        if "yes" in grade.lower():
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant += 1
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            irreleavnt += 1
            continue
    if relevant < irreleavnt:
        web_search = "Yes"
          
    return {"documents": filtered_docs, "web_search": web_search}
    
def web_search2(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    try:
        docs = web_search_tool2.invoke({"query": "site:podatki.gov.pl " + question})
        docs = {"answer": "No web search results found"}
        web_results = Document(page_content=docs)
        documents.append(web_results)
    except:
        documents = []
    return {"documents": documents}

#Graph Edges (validation functions)
def route_question2(state):
    """
    Route question to web search or RAG 

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    route_question = llm_json_mode2.invoke([SystemMessage(content=router_instructions2)] + [HumanMessage(content=state["question"])])
    source = json.loads(route_question.content)['datasource']
    if 'websearch' in source:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif 'vectorstore' in source:
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate2(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if "yes" in web_search.lower():
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question2(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    history = state["history"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 5) # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt2.format(documents=format_docs(documents),history=history, generation=generation.content)
    result = llm_json_mode2.invoke([SystemMessage(content=hallucination_grader_instructions2)] + [HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = json.loads(result.content)['binary_score']

    # Check hallucination
    if "yes" in grade.lower():
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above 
        answer_grader_prompt_formatted = answer_grader_prompt2.format(question=question, history=history, generation=generation.content)
        result = llm_json_mode2.invoke([SystemMessage(content=answer_grader_instructions2)] + [HumanMessage(content=answer_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']
        if "yes" in grade.lower():
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"  
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"


#Creating Graph
workflow2 = StateGraph(GraphState2)

# Define the nodes
workflow2.add_node("websearch", web_search2) # web search
workflow2.add_node("retrieve", retrieve2) # retrieve
workflow2.add_node("grade_documents", grade_documents2) # grade documents
workflow2.add_node("generate", generate2) # generate

# Build graph
workflow2.set_conditional_entry_point(
    route_question2,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow2.add_edge("websearch", "generate")
workflow2.add_edge("retrieve", "grade_documents")
workflow2.add_conditional_edges(
    "grade_documents",
    decide_to_generate2,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow2.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question2,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# memory = MemorySaver()
graph2 = workflow2.compile()
_history = []

def ask_question(question2, max_retries=1):
    global _history
    global graph2
    
    inputs = {"question": question2, "history": _history, "max_retries": max_retries}
    _history.append({"sender":"user", "message":question2})

    inference = []
    for event in graph2.stream(inputs, stream_mode="values"):
        inference.append(event)
        print(event)

    model_output = inference[-1]['generation']
    _history.append({"sender":"you", "message":model_output})
    
    if(len(_history) > 6):
        _history = _history[-4:]

    return model_output

class Message(BaseModel):
  content: str

@app.post("/bartek")
def read_root(message:Message):
  return ask_question(message.content)
