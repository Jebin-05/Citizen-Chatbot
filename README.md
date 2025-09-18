# Citizen-Chatbot

A bilingual chatbot powered by Retrieval-Augmented Generation (RAG) for Tamil Nadu government services and schemes. Supports both English and Tamil languages, providing accurate information on government departments, schemes, and services.

## Features

- **Bilingual Support**: Responds in English or Tamil based on user input.
- **RAG Integration**: Uses vector database for semantic search and retrieval of relevant information.
- **Comprehensive Data**: Includes data from government departments, schemes, and QA pairs.
- **OpenRouter API**: Leverages OpenRouter for AI responses.
- **Modular Design**: Easy to extend with new data sources.

## Installation

### Prerequisites

- Python 3.12 or higher
- Git

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Jebin-05/Citizen-Chatbot.git
   cd Citizen-Chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openrouter_api_key_here
   ```

5. Set up the vector database:
   ```bash
   python setup_rag.py
   ```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

Interact with the chatbot by typing questions in English or Tamil. Type 'quit' to exit.

### Example Queries

- "What are the benefits of the Tamil Nadu government schemes?"
- "தமிழ்நாடு அரசின் திட்டங்களின் நன்மைகள் என்ன?"

## Project Structure

- `chatbot.py`: Main chatbot application
- `setup_rag.py`: Script to set up the RAG vector database
- `document_store.py`: Document processing utilities
- `utils.py`: Helper functions
- `*.json`: Data files for government services and schemes
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore file

## Data Sources

The chatbot uses the following JSON data files:
- `finetune_QA.json`: Q&A pairs for fine-tuning
- `processed_rag_dept.json`: Department-related information
- `processed_rag_services.json`: Service-related data
- `rag_new_scheme.json`: New scheme details
- `tamil_scheme_data.json`: Tamil scheme information

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data sourced from Tamil Nadu government resources
- Powered by OpenRouter API
- Built with LangChain and ChromaDB

## Contact

For questions or support, please open an issue on GitHub.
