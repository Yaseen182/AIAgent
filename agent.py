#!/usr/bin/env python3
"""
Multi-Agent CrewAI System with Groq & Langfuse
FIXED: Using trace() + start_as_current_observation() pattern
"""

import os
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from langfuse import Langfuse

load_dotenv()

# Initialize Langfuse
langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
langfuse = None

if langfuse_enabled:
    try:
        langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        print("✓ Langfuse enabled")
    except Exception as e:
        print(f"⚠️  Langfuse disabled: {e}")
        langfuse_enabled = False

# Use smaller, faster model to avoid rate limits
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5
)

# Tools
search_tool = SerperDevTool()


def create_researcher():
    return Agent(
        role='Researcher',
        goal='Find information',
        backstory='Research expert.',
        tools=[search_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=5
    )


def create_writer():
    return Agent(
        role='Writer',
        goal='Write content',
        backstory='Content writer.',
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=5
    )


def research_and_write(topic: str, content_type: str = "article"):
    """Research and write with proper Langfuse tracking"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Topic: {topic}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Prepare input data
    input_data = {
        "topic": topic,
        "content_type": content_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create agents
    researcher = create_researcher()
    writer = create_writer()
    
    # SHORT task descriptions to save tokens
    research_task = Task(
        description=f"Research: {topic}. Find 3-5 key facts.",
        expected_output="Key facts list",
        agent=researcher
    )
    
    write_task = Task(
        description=f"Write brief {content_type} on {topic}.",
        expected_output=f"Short {content_type}",
        agent=writer,
        context=[research_task]
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False
    )
    
    try:
        print("⏳ Executing (this may take a moment)...")
        
        # Add delay to avoid rate limits
        time.sleep(2)
        
        # Execute with Langfuse tracking
        if langfuse:
            with langfuse.start_as_current_observation(
                as_type="span",
                name=f"research_and_write",
                input=input_data
            ) as span:
                # Execute crew
                result = crew.kickoff()
                result_str = str(result)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Update span with output
                span.update(
                    output={
                        "content": result_str[:800],
                        "length": len(result_str),
                        "status": "success"
                    },
                    metadata={
                        "topic": topic,
                        "content_type": content_type,
                        "execution_time_seconds": duration
                    }
                )
            
            langfuse.flush()
            print("✓ Logged to Langfuse")
            print(f"📊 Dashboard: https://cloud.langfuse.com")
        else:
            # Execute without tracking
            result = crew.kickoff()
            result_str = str(result)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # Log error
        if langfuse:
            try:
                duration = time.time() - start_time
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name=f"error_{topic[:20]}",
                    input=input_data
                ) as span:
                    span.update(
                        output={"error": str(e), "status": "failed"},
                        metadata={
                            "topic": topic,
                            "execution_time_seconds": duration
                        }
                    )
                langfuse.flush()
            except:
                pass
        raise


def quick_research(question: str):
    """Quick research with Langfuse tracking"""
    
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    input_data = {
        "question": question,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    researcher = create_researcher()
    
    task = Task(
        description=f"Answer: {question}. Be brief.",
        expected_output="Short answer",
        agent=researcher
    )
    
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False
    )
    
    try:
        print("⏳ Researching...")
        time.sleep(1)
        
        # Execute with Langfuse tracking
        if langfuse:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="quick_research",
                input=input_data
            ) as span:
                result = crew.kickoff()
                result_str = str(result)
                
                duration = time.time() - start_time
                
                span.update(
                    output={
                        "answer": result_str[:800],
                        "length": len(result_str)
                    },
                    metadata={
                        "question": question,
                        "execution_time_seconds": duration
                    }
                )
            
            langfuse.flush()
            print("✓ Logged to Langfuse")
            print(f"📊 Dashboard: https://cloud.langfuse.com")
        else:
            result = crew.kickoff()
            result_str = str(result)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        if langfuse:
            try:
                duration = time.time() - start_time
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="error_research",
                    input=input_data
                ) as span:
                    span.update(
                        output={"error": str(e)},
                        metadata={"execution_time_seconds": duration}
                    )
                langfuse.flush()
            except:
                pass
        raise


def test_langfuse():
    """Test Langfuse with simple span"""
    if not langfuse:
        print("⚠️  Langfuse not enabled")
        return
    
    try:
        print("Testing Langfuse connection...")
        
        start_time = time.time()
        
        # Use start_as_current_observation with span
        with langfuse.start_as_current_observation(
            as_type="span",
            name="connection_test",
            input={"test": "Connection check", "time": time.strftime("%H:%M:%S")}
        ) as span:
            # Simulate some work
            time.sleep(0.5)
            
            duration = time.time() - start_time
            
            span.update(
                output={"status": "success", "message": "Connection OK"},
                metadata={
                    "test": True,
                    "execution_time_seconds": duration
                }
            )
        
        # Flush to send immediately
        langfuse.flush()
        
        print("✅ Langfuse connection successful!")
        print(f"📊 Check dashboard: https://cloud.langfuse.com")
        print("    Look for span named 'connection_test'")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("""
    ╔════════════════════════════════════════════════╗
    ║      Multi-Agent System with Langfuse         ║
    ║      FIXED: Correct span tracking pattern     ║
    ╚════════════════════════════════════════════════╝
    """)
    
    print(f"Langfuse: {'✅' if langfuse_enabled else '❌'}")
    print("⚠️  Note: Waits added to avoid Groq rate limits\n")
    
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("="*50)
        print("1. Research & Write (takes ~30s)")
        print("2. Quick Research (takes ~15s)")
        print("3. Test Langfuse")
        print("0. Exit")
        print("="*50)
        
        choice = input("\nChoice (0-3): ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye!")
            if langfuse:
                langfuse.flush()
            break
            
        elif choice == "1":
            topic = input("\nTopic: ").strip()
            if not topic:
                print("⚠️  Topic required")
                continue
            
            content_type = input("Type [article]: ").strip() or "article"
            
            try:
                result = research_and_write(topic, content_type)
                print("\n" + "="*50)
                print("✅ RESULT:")
                print("="*50)
                print(result)
            except Exception as e:
                print(f"❌ Failed: {e}")
                
        elif choice == "2":
            question = input("\nQuestion: ").strip()
            if not question:
                print("⚠️  Question required")
                continue
            
            try:
                result = quick_research(question)
                print("\n" + "="*50)
                print("✅ ANSWER:")
                print("="*50)
                print(result)
            except Exception as e:
                print(f"❌ Failed: {e}")
                
        elif choice == "3":
            test_langfuse()
        else:
            print("⚠️  Invalid")


if __name__ == "__main__":
    main()
