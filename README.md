# Content-Engine

- This repository contains the code to run your own custom content engine, please follow the instructions given below strictly in order to get smooth operation of the software:
   
- First make sure you have the ollama installed from the link https://ollama.com/download/windows

- After installation of ollama, for this project we have to install 2 models on the ollama server, the enbedding model "nomic-embed-text" and the LLM "mistral" 

- To install "nomic-embed-text", type "ollama pull nomic-embed-text"
- To install "mistral", type "ollama pull mistral"

- You have to wait for both of them to complete installing, it might take somewhere from 3 minutes to 30 minutes depending on your systems specs

- After installation, you can check the models in your ollama by typing: "ollama list"

- After you verify your models, its time to launch the server by using the command: "ollama serve"

- Now that the server is running, we can start with our code
   - Clone my repository into your local system using "git clone https://github.com/hydracsnova13/Content-Engine.git"
   
   - After cloning it, now we will add the pdf files that you want to query into the data folder in the same directory
 
   - Before we start to execute our project we make a virtual environment to work in so any modifications are local and don't effect our system using the command for windows "virtualenv <environment_name>"
 
   - If the virtualenv doest work, make sure to install virtualenv using "pip install virtualenv"
 
   - Also before everythingq, make sure your python is installed and updated to the latest version
 
   - After the virtual environment is installed, make sure to activate it by going to the scripts folder on your environment and running activate
 
   - Then go to the directory where the code files are there and install the dependencies using the command: "pip install -r requirements.txt"
 
   - Then run the python code to add the document clusters to the chroma database, where the vectors are stored, we have to run the populate_database.py, "python populate_database.py", if you want to clear the database and remove any existing documents and repopulate with new content, you can use "python populate_database.py --reset"
 
   - This will take some time even though 4 threads are executing the process
 
   - Upon completion, now your chroma database has been populated with the new documents extracted from the pdf's provided in the data directory
 
   - Now we run the streamlit app "app.py" which will start the streamlit app to run the query process and retrieval process, run the command: "streamlit run app.py"
 
   - This will open the chat interface and now you can freely query your pdf's with the help of the chat interface
 
   - note, that the time of reply by the server is entirely dependent on your own system's hardware configurations
