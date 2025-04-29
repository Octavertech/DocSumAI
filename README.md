# DocSumAI

## AI-Powered Document Summarization Engine

DocSumAI is an advanced document summarization tool leveraging generative AI to transform lengthy documents into concise, accurate summaries while preserving key information and context.

## Key Features

- **Multi-Format Support**: Process PDFs, Word documents, text files, HTML pages, and more
- **Intelligent Summarization**: Uses state-of-the-art LLMs to generate contextually aware summaries
- **Customizable Output**: Adjust summary length, focus areas, and tone based on your needs
- **Multilingual Capability**: Summarize documents in multiple languages
- **Information Extraction**: Automatically identify and highlight key entities, dates, figures, and critical points
- **Batch Processing**: Summarize multiple documents simultaneously
- **API Integration**: Easily integrate with existing workflows and applications

## Use Cases

- Research acceleration for academics and professionals
- Legal document analysis and brief creation
- Business intelligence report summarization
- News and content curation
- Technical documentation simplification
- Meeting notes and transcript summarization

## Technical Stack

- Python-based backend with FastAPI
- HuggingFace Transformers
- FastAPI – API framework
- React frontend with responsive design
- Docker support for easy deployment
- Uvicorn – ASGI server
- Pydantic – Data validation
- pdfplumber – PDF text extraction
- pdfminer.six – Advanced PDF parsing
- pymupdf – Lightweight PDF handling
- python-docx – DOCX text extraction
- nltk – Natural Language Processing (NER, Sentiment Analysis)
- transformers – HuggingFace transformers (for future extensions if needed)
- torch – PyTorch backend for models
- openai – (Optional) OpenAI model usage (future-proof)
- google-generativeai – Google Gemini API
- dotenv – Environment variable management
- requests – HTTP requests
- services – Custom service layer
- python-multipart – Handling file uploads
- Leverages Transformer-based language models
- Document processing via PyPDF2, docx2txt, and BeautifulSoup


## License

This project is licensed under the OctaverTech LLC.
