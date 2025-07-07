from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Assuming retriever is properly defined

# Initialize the LLM model
model = OllamaLLM(model="llama3.2:1b")

# Define a new template for educational content
template = """
        You are an indian expert in answering educational questions based on the provided content.

       you are expert in answering educational questions based on the provided content. 
       student ae from india and they are asking questions in english.
        Here are some relevant educational materials: {reviews}

        Here is the question to answer: {question}
    """


# Create the prompt using the defined template
prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt with the model to form the chain
chain = prompt | model

# # Start the interactive loop
# while True:
#     print("\n\n-------------------------------")
#     question = input("Ask your question (q to quit): ")
#     print("\n\n")
    
#     # Exit condition if user types 'q'
#     if question == "q":
#         break
    
#     # Get the relevant reviews (educational content) from the retriever
#     reviews = retriever.invoke(question)
    
#     # Generate the result using the chain
#     result = chain.invoke({"reviews": reviews, "question": question})
    
#     # Print the result (answer)
#     print(result)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS to allow requests from the browser (important!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class Question(BaseModel):
    que: str


# POST endpoint
@app.get("/home")
async def home():
    
        return {"answer": "result"}

# POST endpoint
@app.post("/")
async def get_answer_from_pdf(question: Question):
    try:
        # Extract the plain string
        query = question.que

        # Use string, not the Pydantic model
        reviews = retriever.invoke(query)

        # Pass plain dict values to chain
        result = chain.invoke({
            "reviews": reviews,
            "question": query
        })

        print(result)
    
        return {"answer": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


