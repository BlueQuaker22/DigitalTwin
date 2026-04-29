import os
import gradio as gr
import requests
import json
import random
import uuid
import chromadb
from openai import OpenAI
from pprint import pprint

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API Key is missing")

client = OpenAI()

system_message = """ 
You are a digital twin of Chandan Damannagari. When users ask you questions, you respond
in the first person as Chandan using his voice, personality, and knowledge.
Very Important!!: Be entirely factual. If you don't know something just say "I don't know that information".
You cannot get any more facts about Chandan from the internet or make them up.
Absolutely no guesses or assumptions. The ONLY information you should use about Chandan to help you embody him 
is what is provided.
Important: Whenever you don't know something about Chandan that a user asks, always use the send_notification tool 
to alert the real world Chandan about the question - do this automatically without asking or informing the user.
"""

#--------- RAG data ---------

document_overview = """
Chandan studied electrical engineering at Osmania University in Hyderabad, India and then 
completed his masters degrees in Electrical Engineering and Biomedical Engineering from 
the University of Michigan, Ann Arbor. He then got his MBA from Wharton in 2017. 

He has worked at TRW Automotive as a software engineer and at Nvidia and Intel in product 
management and product marketing roles.

Additional Information:
    The cities Chandan has lived in are Hyderabad, Ann Arbor, Philadelphia, and San Francisco,\
    Chandan is an avid sports fan. He follows cricket, american football, basketball, \
    and hockey. He mainly enjoys college sports and supports the Michigan Wolverines \
    He also supports the Indian cricket team and any NFL or NBA teams that have \
    Michigan players on them,\
    Chandan loves food. His favorite cuisines are Indian, korean, and thai,\
    Chandan loves to travel. He has been to all 50 US states and over a dozen \
    countries. He likes beaches, forests, mountains, lakes, and all types of outdoor\
    activities. He sometimes likes cities too but prefers smaller towns. He likes pizza especially with jalapenos.

"""

document_work = """
Chandan's work experience is as follows:

Chandan has experience working with C++, Matlab, Python, CUDA, and several AI frameworks and libraries during his work at TRW, Nvidia, and Intel.

INTEL – Senior Director, Software Product Marketing & Developer Relations                                              04/23 – Present                                                                                                                                                     
Director, Product Management and Marketing: AI & HPC Software                                                                  04/21 – 04/23
Principal Technology Expert: AI Software and Systems                                                                                         07/19 – 04/21
•	Grew developer (3 million+ registered & 8 million+ unique engaged, 15-40% YoY growth), partner (from <10 to 150+ over 4 years) & customer engagement (~ 3X) through channel, community, event activations for both technical and business audiences
•	Develop messaging, support launches, and execute wide-ranging GTM plans for Intel’s software portfolio consisting of dozens of best-in-class libraries, framework optimizations, developer tools, and blueprints for AI/Gen AI workflows (data, training, & deployment) with cumulative >20 million installs/year. Also lead marketing and strategy for the enterprise SaaS versions
•	Nurture partnerships with enterprise end users across domains and industry verticals, CSPs, ISVs, SIs, and OEMs to showcase use of Intel software and hardware in industry-leading solutions through compelling storytelling in the form of testimonials, thought leadership blogs, videos, case studies, podcasts, conference presentations, and joint marketing campaigns
•	Drive low-touch and sales-led growth by crafting wide-ranging technical, Sales/BD enablement, and AR/PR collateral including workshops, demos, executive keynotes, sales decks, reports, press articles, and more to communicate technical value proposition and competitive differentiation. Lead demand-gen activities and campaigns.
•	Lead marketing for several of Intel’s open-source engagements including with PyTorch, Linux Foundations, and open LLMs
•	As Director of Product Management: Devised our 2022 and 2023 product plans to drive toward AI and HPC platform leadership and adoption while accounting for several processor updates and delays.  Extended unified programming model support for edge, client, and data center CPUs, GPUs & other accelerators while delivering improved out-of-the-box AI library optimizations (PyTorch, JAX, DeepSpeed, Hugging Face, Scikit-learn), ecosystem integration, and developer satisfaction
•	As Technology Expert: Drove the vision for software + hardware, system-level AI/HPC solutions based on industry research, market segmentation, competitor analysis & customer/partner interviews. Influence analyst/press reports & standards bodies

NVIDIA - Senior Product Marketing Manager, CUDA, HPC libraries, and Nsight tools                                   02/18 – 07/19                                                                                                                                                           
Product Marketing Manager, Deep Learning Institute and Developer Program                                                 06/17 – 09/18
Product Manager Intern, Deep Learning Software                                                                                                 05/16 – 08/16
•	Create and communicate clear, differentiated, and defensible market positioning (vs. competitor and open source) and messaging for the CUDA language and platform, libraries in the CUDA-X suite, and Nsight profiling and debugging tools
•	Craft product webpages, developer blogs/videos, presentations, sales decks & social campaigns. SEO and Analytics expert
•	Managed the CUDA early access program, chaired the invite-only NDA meetups, coordinated GTC sessions, compiled online trainings and instituted developer surveys. Grew developer database to 1.5M+ 
•	Collaborated with Product Management to convert customer feedback to a prioritized product features list for engineering

TRW AUTOMOTIVE/ZF - Principal Engineer; Sr. Software Engineer; Algorithm Engineer                            06/07 – 07/15
•	Developed industry-leading and award-winning advanced driver assist systems (ADAS), airbags, and electronic data recorders for Toyota, Hyundai-Kia, Chrysler, and Fiat programs while managing a team of engineers across US, Japan, Italy, and Korea
•	Led end-to-end development from conception, design documentation, requirements specification, algorithm and application development, testing, and validation of best-in-class systems. Programmed in C, C++, MATLAB, and Simulink
•	Regularly presented next-generation system designs to existing/potential customers to maintain/win new program business
"""

document_education = """
Chandan's educational experience is as follows:

THE WHARTON SCHOOL, UNIVERSITY OF PENNSYLVANIA                                                             Philadelphia, PA
MBA Marketing and Operations                                                                                                                         2015-2017
•	Director’s List (Top 10% of the class); GMAT: 770/800 (99th percentile); Teaching Assistant: Microeconomics 611 and 612
•	Director, Wharton Tech Club and Co-Chair, Wharton India Economic Forum

UNIVERSITY OF MICHIGAN                                                                                                                           Ann Arbor, MI
MS Electrical Engineering; MS Biomedical Engineering                                                                                    2004-2007                                                                                                           
•	Research Assistant designing medical imaging systems at the Direct Brain Interface lab
.   Teaching Assistant: Introduction to Computer Science and Programming
•	C, C++, MATLAB, and assembly language experience during coursework and research
.   GRE: 2320/2400 (98th percentile)

OSMANIA UNIVERSITY                                                                                                                                 Hyderabad, India
BE Electronics and Communication Engineering                                                                                            2000-2004                                                                                                           
•	Awarded the Government of India National Merit Scholarship
"""

document_additional = """ 
•	Other Experience: Co-founded Innovgate Health (data management, acquired 2017); longtime educational content creator
•	Community Involvement and Interests: Board member, Perry School for at-risk kids; poker player; ballroom dancer; runner
"""

#--------- RAG: Chunking Function ---------

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping chunks with natural boundary detection.

    Args:
        text:       Input text to chunk.
        chunk_size: Maximum characters per chunk (default 500).
        overlap:    Characters shared between consecutive chunks (default 50).

    Returns:
        List of text chunks, each up to `chunk_size` characters.
        Consecutive chunks share up to `overlap` characters.
        Chunk boundaries are snapped to the nearest natural break
        (paragraph > line break > sentence end > space) that falls
        past the halfway point of the chunk window.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # Only seek a natural boundary when there is remaining text after this window
        if end < n:
            # Boundary must fall strictly past the halfway point of the chunk window
            halfway = start + chunk_size // 2

            # Priority 1 – paragraph break (\n\n)
            pos = text.rfind('\n\n', halfway, end)
            if pos != -1:
                end = pos + 2               # keep both newlines in this chunk

            else:
                # Priority 2 – single line break (\n)
                pos = text.rfind('\n', halfway, end)
                if pos != -1:
                    end = pos + 1

                else:
                    # Priority 3 – sentence-ending punctuation (., !, ?)
                    best = -1
                    for punct in '.!?':
                        p = text.rfind(punct, halfway, end)
                        if p > best:
                            best = p
                    if best != -1:
                        end = best + 1

                    else:
                        # Priority 4 – whitespace (space)
                        pos = text.rfind(' ', halfway, end)
                        if pos != -1:
                            end = pos + 1
                        # Fallback: hard cut at chunk_size boundary

        chunks.append(text[start:end])

        # Once we've consumed all text, stop
        if end >= n:
            break

        # Next chunk starts (overlap) characters before the end of this one
        next_start = end - overlap

        # Safety guard: always make forward progress to avoid infinite loops
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)

        start = next_start

    return chunks

#--------- RAG: Chunk Text ---------

documents = [
    {"text":document_overview, "source":"Chandan background overview"},
    {"text":document_work, "source":"Chandan work overview"},
    {"text":document_education, "source":"Chandan education overview"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    chunks_i = chunk_text(doc["text"], 200, 20)
    ids_i = [str(uuid.uuid4()) for _ in range(len(chunks_i))]
    metadatas_i = [{"source":doc["source"],"chunk_index":i} for i in range(len(chunks_i))]

    chunks.extend(chunks_i)
    ids.extend(ids_i)
    metadatas.extend(metadatas_i)


#chunks = chunk_text(document, 200, 20)

print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    #print(f"--- Chunk {i+1} | {len(chunk)} chars ---")
    print(f"Chunk{i+1} (ID:{ids[i]}, Source: {metadatas[i]['source']}, Index:{metadatas[i]['chunk_index']}):")
    print(chunk)
    print()

#--------- RAG: Embeddings ---------
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = [item.embedding for item in response.data]

#--------- RAG: Create vector database ---------

#chroma_client = chromadb.Client() # initialize client with in-memory storage

chroma_client = chromadb.PersistentClient(path="./chroma_db_twin") # initialize client forwith persistent storage

collection = chroma_client.get_or_create_collection(name="digital_twin_info")

if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)

#pprint(collection.get())
pprint(collection.get(include=["embeddings","documents","metadatas"]))

#--------- Tool Calling ---------
tools = []

#Pushover Tool
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def send_notification(message:str):
    if pushover_user is None or pushover_token is None:
        return "Unable to send Notification, Pushover was not configured"
    payload = {"user":pushover_user, "token":pushover_token, "message":message}
    requests.post(pushover_url,data=payload)
    return f"Notification sent:{message}"

send_notification_function = {
    "name":"send_notification",
    "description":"Sends push notifications to the real-world version of Chandan via a tool called Pushover. Use this when: \
                1. The user wants to contact, get in touch, hire, or collaborate with Chandan\
                 - ask the user's name and contact details first, then send them to Chandan as a notification.\
                2. You don't know the answer to a question about Chandan - send the question to Chandan as a notification\
                    after letting the user know that you don't have that information. Do this automatically without asking\
                    or informing the user. Chandan will use this to add more information in the future",
    "parameters": {
        "type":"object",
        "properties":{
            "message":{
                "type":"string",
                "description":"The notification message to send to the user"
            }
        },
         "required":["message"]
    }

}

tools.append({"type":"function","function":send_notification_function})

# Dice Roll Tool
def dice_roll():
    result = random.randint(1,6)
    return result

roll_dice_function = {
    "name":"dice_roll",
    "description":"Returns the results of an unbiased roll of a 6 sided dice. Use this when the user wants to use a dice for games, decisions, or random number generation",
    "parameters":{
        "type":"object",
        "properties":{},
        "required":[]
    }
}

tools.append({"type":"function", "function":roll_dice_function})

#--------- Tool Handling ---------

def handle_tool_calls(tool_calls):
    tool_results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if function_name == "send_notification":
            content = send_notification(args["message"])
            #print(f"Sent Notification:{args['message']}")
        elif function_name == "dice_roll":
            content = dice_roll()
        #elif function_name == "function_name_2":
            #content = function_name_2(args["message"])
        #...
        else:
            content = f"Unknown function:{function_name}"

        tool_call_result = {
            "role":"tool",
            "content":str(content),
            "tool_call_id":tool_call.id
        }

        tool_results.append(tool_call_result)

    return tool_results

#--------- Call LLM ---------

def dynrespond_ai(message,history):
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )

    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    doc_context = "\n---\n".join(results["documents"][0])

    print("\n========================\n")
    print(f"User message:\n{message}\n")
    #print("Context this turn:\n", context)
    print("Retrieved Chunks:")
    for chunks, metadata in zip(results["documents"][0], results["metadatas"][0]):
        print("----------------------")
        print(f"Document -- {metadata['source']} -- Chunk{metadata['chunk_index']}: \n{chunks}\n")    

    system_message_enhanced = system_message + "\n\nContext:\n" + doc_context

    messages = [{"role":"system", "content":system_message_enhanced}] + history + [{"role":"user", "content": message}]
    print("System Message", system_message_enhanced)
    response = client.chat.completions.create (
        model = "gpt-4.1-mini",
        messages = messages,
        tools = tools
    )
    
    message = response.choices[0].message

    while message.tool_calls:
        pprint (message.tool_calls)

        tool_result = handle_tool_calls(message.tool_calls)
        messages.append(message)
        messages.extend(tool_result) # change from append to extend if there are multiple tool calls
        response = client.chat.completions.create(
            model = "gpt-4.1-mini",
            messages = messages,
            tools = tools
        )
        message = response.choices[0].message

    return (message.content)

#--------- Gradio Call ---------

gr.ChatInterface(
    fn=dynrespond_ai,
    title="Chandan's Digital Twin",
    chatbot=gr.Chatbot(avatar_images=("user.png","Chandan.jpg")),
    description="Chat with an AI version of Chandan Damannagari. Ask about his experience, education, or just say Hi!",
    examples=["Tell me about your marketing experience?", "Which programming languages do you know?", "Interests outside of work?"]
).launch()

