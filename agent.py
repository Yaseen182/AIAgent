import time
import re
from langfuse import get_client, Langfuse
import math
import random

# ---- Initialize Langfuse ----
lf = Langfuse(
    public_key="pk-lf-a9832822-04ea-4367-9eb1-92ab6029ef44",
    secret_key="sk-lf-088016ad-6d96-4814-aa57-c051388c4575",
    base_url="https://cloud.langfuse.com"
)
lf = get_client()

# ---- Tools ----
def calculator(question: str):
    """Calculate math expression inside the question"""
    try:
        expr_match = re.search(r'[0-9\+\-\*/\(\)\s\.]+', question)
        if expr_match:
            expression = expr_match.group(0)
            return str(eval(expression))
        else:
            return "No valid expression found."
    except Exception as e:
        return f"Error: {e}"

def random_joke(_):
    """Return a random joke to entertain the user."""
    jokes = [
        "Why did the computer go to the doctor? It caught a virus!",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why did the AI cross the road? To optimize the other side!"
    ]
    return random.choice(jokes)

def word_count(text):
    """Count the number of words in the given text."""
    return f"The text contains {len(text.split())} words."

tools = {
    "calculator": calculator,
    "random_joke": random_joke,
    "word_count": word_count
}

tool_descriptions = {
    "calculator": "Use this tool to calculate any math expression.",
    "random_joke": "Use this tool to tell a funny joke.",
    "word_count": "Use this tool to count the number of words in a text."
}




def choose_tool(question: str):
    question_lower = question.lower()
    if re.search(r'[0-9\+\-\*/\(\)]', question):
        return "calculator"
    elif "joke" in question_lower:
        return "random_joke"
    elif "word count" in question_lower or "how many words" in question_lower:
        return "word_count"
    else:
        return None


def ai_agent(question: str):
    start_time = time.time()
    
    with lf.start_as_current_observation(
        as_type="span",
        name="auto-tool-agent",
        input={"question": question}
    ) as span:
        
        selected_tool = choose_tool(question)
        
        if selected_tool:
            answer = tools[selected_tool](question)
        elif "ai" in question.lower():
            answer = "AI stands for Artificial Intelligence. It can perform tasks using multiple tools like calculator, text analysis, and more."
        else:
            answer = "Sorry, I don't understand your question."
        
        duration = time.time() - start_time
        
        span.update(
            output={"answer": answer, "selected_tool": selected_tool},
            metadata={"execution_time_seconds": duration}
        )

    lf.flush()
    return answer

if __name__ == "__main__":
    question = input("Enter your question: ")
    print(f"Q: {question}")
    print(f"A: {ai_agent(question)}\n")
