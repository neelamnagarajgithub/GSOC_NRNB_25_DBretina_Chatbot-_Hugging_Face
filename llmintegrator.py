from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="nagaraj",
    password="password"
)

# Import movie data into Neo4j
movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""
graph.query(movies_query)
graph.refresh_schema()
logger.info("Graph schema: %s", graph.schema)

# Load Hugging Face model
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Create Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    max_length=1024,
    truncation=True,  
    temperature=0.7,   
    pad_token_id=tokenizer.eos_token_id  
)


# Initialize LLM for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Use Neo4jGraph directly in GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)

def execute_query(query: str):
    result = chain.invoke({"query": query})
    print("Raw Response:", result)

    if isinstance(result, dict):
        cypher_query = result.get("intermediate_steps", "No Cypher Generated")
        final_answer = result.get("result", "No Answer Generated")

        print("Generated Cypher:", cypher_query)
        print("Final Answer:", final_answer)

        return {"cypher": cypher_query, "answer": final_answer}

    return result

