# Plagiarism Detector - Semantic Similarity Analyzer

A modern web application that detects plagiarism using semantic similarity analysis with multiple state-of-the-art embedding models.

## Features

- **Multiple Embedding Models**: Compare results using:
  - Sentence Transformers (all-MiniLM-L6-v2)
  - OpenAI text-embedding-3-small
  - Nomic Embed Text v1.5
- **Dynamic Text Input**: Add unlimited text boxes for comparison
- **Similarity Matrix**: Visual representation of pairwise similarities
- **Clone Detection**: Automatically identifies texts with >80% similarity
- **Model Comparison**: Side-by-side comparison of different embedding models
- **Responsive Design**: Works on desktop and mobile devices

## How It Works

### Embedding Models
1. **Sentence Transformers**: Lightweight and fast, good for general-purpose tasks
2. **OpenAI Embeddings**: Commercial model with excellent performance
3. **Nomic Embed**: Open-source model that outperforms OpenAI on many benchmarks

### Similarity Detection
- Texts are converted to numerical vectors (embeddings) that capture semantic meaning
- Cosine similarity measures the angle between vectors (0-1 scale)
- Threshold-based detection flags texts with >80% similarity as potential clones
- Handles paraphrasing and meaning-based similarities, not just word matches

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd plagiarism-detector
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional):
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
NOMIC_API_KEY=your_nomic_api_key_here
```

## Usage

1. **Start the application**:
```bash
python app.py
```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Add your texts** using the dynamic input boxes

4. **Select embedding models** you want to compare

5. **Click "Analyze for Plagiarism"** to run the analysis

## Model Information

### Sentence Transformers (all-MiniLM-L6-v2)
- **Type**: Open Source
- **Context Length**: 512 tokens
- **Dimensions**: 384
- **Best for**: Fast processing, general-purpose tasks

### OpenAI text-embedding-3-small
- **Type**: Commercial (requires API key)
- **Context Length**: 8,191 tokens
- **Dimensions**: 1,536
- **Best for**: High accuracy, production use

### Nomic Embed Text v1.5
- **Type**: Open Source
- **Context Length**: 8,192 tokens
- **Dimensions**: 768
- **Best for**: State-of-the-art performance, outperforms OpenAI on many benchmarks

## API Endpoints

- `GET /` - Main application interface
- `POST /analyze` - Analyze texts for plagiarism
- `GET /health` - Health check and model status

## Example Use Cases

1. **Academic Integrity**: Check student submissions for potential plagiarism
2. **Content Review**: Identify duplicate or similar content across documents
3. **Research**: Compare different embedding models' performance
4. **Legal Documents**: Find similar clauses or passages

## Similarity Thresholds

- **>80%**: Potential clone (flagged as suspicious)
- **60-80%**: High similarity (review recommended)
- **<60%**: Low similarity (likely original)

## Technical Architecture

```
User Interface (HTML/CSS/JS)
         ↓
Flask Web Application
         ↓
PlagiarismDetector Class
         ↓
Multiple Embedding Models
         ↓
Cosine Similarity Calculation
         ↓
Results with Similarity Matrix
```

## Dependencies

- Flask 3.0.0 - Web framework
- sentence-transformers 2.2.2 - Sentence embeddings
- openai 1.3.0 - OpenAI API client
- numpy 1.24.3 - Numerical computing
- scikit-learn 1.3.0 - Machine learning utilities
- pandas 2.0.3 - Data manipulation
- plotly 5.17.0 - Data visualization
- python-dotenv 1.0.0 - Environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure you have sufficient disk space and internet connection
2. **API Key Error**: Check that your API keys are correctly set in the `.env` file
3. **Memory Issues**: Use smaller batch sizes or fewer models simultaneously

### Performance Tips

- Use Sentence Transformers for fastest processing
- OpenAI embeddings require API calls (may be slower)
- Nomic embeddings offer best balance of speed and accuracy

## Future Enhancements

- [ ] Support for more embedding models
- [ ] Batch processing for multiple files
- [ ] Export results to CSV/PDF
- [ ] Advanced similarity threshold customization
- [ ] Multi-language support
- [ ] Integration with document management systems 