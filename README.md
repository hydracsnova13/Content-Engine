# Content-Engine

This repository contains the code to run your own custom content engine. Please follow the instructions below to ensure smooth operation of the software.

## Prerequisites

1. **Install Ollama:**
   - Download and install Ollama from [this link](https://ollama.com/download/windows).

2. **Install Models on Ollama:**
   - **Embedding Model:** Install the "nomic-embed-text" model by running:
     ```bash
     ollama pull nomic-embed-text
     ```
   - **LLM Model:** Install the "mistral" model by running:
     ```bash
     ollama pull mistral
     ```
   - Wait for both installations to complete. This may take between 3 to 30 minutes depending on your system's specifications.

3. **Verify Installed Models:**
   - Check the installed models by running:
     ```bash
     ollama list
     ```

4. **Launch Ollama Server:**
   - Start the server by running:
     ```bash
     ollama serve
     ```

## Getting Started

1. **Clone the Repository:**
   - Clone this repository into your local system:
     ```bash
     git clone https://github.com/hydracsnova13/Content-Engine.git
     ```

2. **Add PDF Files:**
   - Add the PDF files that you want to query into the `data` folder in the cloned directory.

3. **Set Up Virtual Environment:**
   - Create a virtual environment to work in, ensuring modifications are local:
     ```bash
     virtualenv <environment_name>
     ```
   - If `virtualenv` is not installed, install it using:
     ```bash
     pip install virtualenv
     ```
   - Ensure Python is installed and updated to the latest version.

4. **Activate Virtual Environment:**
   - Activate the virtual environment by navigating to the `Scripts` folder within your environment and running `activate`.

5. **Install Dependencies:**
   - Navigate to the directory where the code files are located and install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Running the Project

1. **Populate Chroma Database:**
   - Run the script to add document clusters to the Chroma database:
     ```bash
     python populate_database.py
     ```
   - To clear the database and repopulate with new content, run:
     ```bash
     python populate_database.py --reset
     ```
   - This process might take some time as it utilizes 4 threads.

2. **Run the Streamlit App:**
   - Start the Streamlit app to initiate the query and retrieval process:
     ```bash
     streamlit run app.py
     ```
   - This will open the chat interface. You can now freely query your PDFs using the chat interface.

**Note:** The response time of the server is entirely dependent on your system's hardware configurations.

---

Feel free to contribute to this project by forking the repository and submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.
