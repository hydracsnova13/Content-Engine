import streamlit as st
import aiohttp
import asyncio
import json
import re
import logging
from query_data import get_condensed_context  # Import the context retrieval function

logging.basicConfig(level=logging.INFO)

# Define the prompt template with a character limit instruction
CHARACTER_LIMIT = 500  # Set your desired character limit here
PROMPT_TEMPLATE = """
Answer the question based only on the following context in an optimal manner. Ensure the response does not exceed {char_limit} characters:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

async def invoke_model_async(prompt):
    url = "http://localhost:11500/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral",
        "prompt": prompt
    }

    json_data = json.dumps(data)
    logging.info(f"Raw JSON data to be sent: {json_data}")

    response_text = ""
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        try:
            async with session.post(url, headers=headers, data=json_data) as response:
                logging.info("Sent request to Mistral model with prompt:")
                logging.info(prompt)
                logging.info("Waiting for response from Mistral model...")

                async for line in response.content:
                    line_data = line.decode('utf-8')
                    logging.info(f"Received chunk: {line_data}")

                    # Extract the "response" part from the JSON chunk
                    match = re.search(r'"response":"(.*?)"', line_data)
                    if match:
                        chunk_text = match.group(1).replace("\\n", "\n").replace("\\t", "\t")
                        response_text += chunk_text

                logging.info("Full response received:")
                logging.info(response_text)

                return response_text

        except aiohttp.ClientError as e:
            logging.error(f"ClientError occurred: {e}")
            return f"ClientError: {e}"
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            return f"Unexpected error: {e}"

def main():
    st.title("PDF Guru Interface")

    # Display chat history
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for i, message in enumerate(st.session_state.chat_history):
            if message.startswith("😎: "):
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin: 5px;">
                        <div style="background-color: black; border: 2px solid lightgreen; padding: 10px; border-radius: 10px; max-width: 60%;">
                            {message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin: 5px;">
                        <div style="background-color: black; border: 2px solid red; padding: 10px; border-radius: 10px; max-width: 60%;">
                            {message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    query_text = st.text_area("Message:", key=f"message_input_{len(st.session_state.chat_history)}", disabled=st.session_state.processing, height=80)

    if st.button("Send", disabled=st.session_state.processing):
        query_text = query_text.strip()
        if query_text:
            st.session_state.processing = True

            # Retrieve condensed context
            context_text = get_condensed_context(query_text)

            # Create the prompt with the condensed context
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text, char_limit=CHARACTER_LIMIT)

            async def run_query():
                response = await invoke_model_async(prompt)
                st.session_state.chat_history.append(f"😎:  {query_text}")
                st.session_state.chat_history.append(f"🤖:  {response}")  # Append the bot response
                st.session_state.processing = False
                st.rerun()

            # Use asyncio.run() only if no event loop is running
            try:
                asyncio.get_running_loop()
                asyncio.create_task(run_query())
            except RuntimeError:
                asyncio.run(run_query())

            st.rerun()  # Immediately update UI after setting processing state and sending request

if __name__ == "__main__":
    main()
