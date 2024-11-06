### Prerequisites
1. Python 3.9+ (less than 3.13 for dependency purposes)
2. Git 
3. Text Editor 

## Download [Ollama](https://ollama.com/download)

## Download code
Enter directory of code

## Dependencies

To Create virtual environment (Optional):
On MAC/Linux :
```
python3 -m venv summit
source summit/bin/activate
```
On Windows :
```
python3 -m venv summit
summit\Scripts\activate
```
To run this application, you need to have the following dependencies installed:

Ollama Dependencies -
```
ollama pull llama3
ollama pull nomic-embed-text
```
Python Dependencies - 
```
pip install python-docx langchain langchain_community langchain_core streamlit chromadb
```
## Local Chatbot

You can use the below command to have a local chat bot
```
ollama run llama3
```


## Usage

- Run the application using the following command:
```
streamlit run app.py
```

- Enter your question in the text input field and click the "Query Documents" button to get the answer based on the context documents.


## Resources
This application uses the following libraries and tools:

- [Streamlit](https://streamlit.io/) for the web application framework.
- [LangChain](https://langchain.com/) for the language model and document processing.
- [python-docx](https://python-docx.readthedocs.io/) for DOCX file processing.
- [Chroma](https://www.trychroma.com/) for the vector database.
- [Ollama](https://www.anthropic.com/models) for the language model.


## Further learning

- [Prompt Leaks](https://github.com/jujumilk3/leaked-system-prompts)
- [Langchain](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/)
- [Motific](https://motific.ai)
- [Mitre Framework](https://atlas.mitre.org/studies)
- [OWASP Top 10 Vulnerabilities](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_1.pdf)
- [LLM fundamental knowledge](https://github.com/mlabonne/llm-course)
- [ROME](https://github.com/kmeng01/rome)
- [Supply Chain Attack](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/)
- [Attenton is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
