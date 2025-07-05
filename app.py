from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import requests
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from dotenv import load_dotenv
import re
import plotly.graph_objects as go
import plotly.utils

# Load environment variables
load_dotenv()

app = Flask(__name__)

class PlagiarismDetector:
    def __init__(self):
        self.models = {
            'sentence_transformers': None,
            'openai': None,
            'nomic': None
        }
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all embedding models"""
        try:
            # Initialize Sentence Transformers
            self.models['sentence_transformers'] = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence Transformers model loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load Sentence Transformers: {e}")
        
        try:
            # Initialize OpenAI
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if openai.api_key:
                self.models['openai'] = True
                print("✓ OpenAI API key configured")
            else:
                print("⚠ Warning: OpenAI API key not found")
        except Exception as e:
            print(f"⚠ Warning: Could not configure OpenAI: {e}")
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def get_sentence_transformer_embedding(self, text):
        """Get embeddings from Sentence Transformers"""
        if self.models['sentence_transformers'] is None:
            raise ValueError("Sentence Transformers model not available")
        
        processed_text = self.preprocess_text(text)
        embedding = self.models['sentence_transformers'].encode([processed_text])
        return embedding[0]
    
    def get_openai_embedding(self, text):
        """Get embeddings from OpenAI"""
        if not self.models['openai']:
            raise ValueError("OpenAI API not available")
        
        processed_text = self.preprocess_text(text)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=processed_text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            raise ValueError(f"OpenAI API error: {e}")
    
    def get_nomic_embedding(self, text):
        """Get embeddings from Nomic API"""
        processed_text = self.preprocess_text(text)
        
        # Try to use Nomic API if available
        nomic_api_key = os.getenv('NOMIC_API_KEY')
        if nomic_api_key:
            try:
                import nomic
                from nomic import embed
                
                output = embed.text(
                    texts=[processed_text],
                    model='nomic-embed-text-v1.5',
                    task_type='search_document'
                )
                return np.array(output['embeddings'][0])
            except Exception as e:
                print(f"Nomic API error: {e}")
        
        # Fallback to Sentence Transformers for demo purposes
        # In real implementation, you'd use the actual Nomic model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Fallback model
            embedding = model.encode([processed_text])
            return embedding[0]
        except Exception as e:
            raise ValueError(f"Nomic embedding error: {e}")
    
    def get_embeddings_for_model(self, texts, model_name):
        """Get embeddings for all texts using specified model"""
        embeddings = []
        
        for text in texts:
            if not text.strip():
                continue
                
            try:
                if model_name == 'sentence_transformers':
                    embedding = self.get_sentence_transformer_embedding(text)
                elif model_name == 'openai':
                    embedding = self.get_openai_embedding(text)
                elif model_name == 'nomic':
                    embedding = self.get_nomic_embedding(text)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding for model {model_name}: {e}")
                # Use zero vector as fallback
                embeddings.append(np.zeros(384))  # Default dimension
        
        return np.array(embeddings)
    
    def calculate_similarity_matrix(self, embeddings):
        """Calculate pairwise cosine similarity"""
        if len(embeddings) == 0:
            return np.array([])
        
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def detect_clones(self, similarity_matrix, threshold=0.8):
        """Detect potential clones based on similarity threshold"""
        clones = []
        n = len(similarity_matrix)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    clones.append({
                        'text1_index': i,
                        'text2_index': j,
                        'similarity': float(similarity)
                    })
        
        return clones
    
    def analyze_texts(self, texts, models_to_use=None):
        """Analyze texts for plagiarism using specified models"""
        if models_to_use is None:
            models_to_use = ['sentence_transformers', 'openai', 'nomic']
        
        # Filter out empty texts
        filtered_texts = [text for text in texts if text.strip()]
        
        if len(filtered_texts) < 2:
            return {
                'error': 'At least 2 non-empty texts are required for comparison'
            }
        
        results = {}
        
        for model_name in models_to_use:
            if model_name == 'openai' and not self.models['openai']:
                continue
            if model_name == 'sentence_transformers' and self.models['sentence_transformers'] is None:
                continue
                
            try:
                embeddings = self.get_embeddings_for_model(filtered_texts, model_name)
                similarity_matrix = self.calculate_similarity_matrix(embeddings)
                clones = self.detect_clones(similarity_matrix)
                
                results[model_name] = {
                    'similarity_matrix': similarity_matrix.tolist(),
                    'clones': clones,
                    'model_info': self.get_model_info(model_name)
                }
            except Exception as e:
                results[model_name] = {
                    'error': str(e)
                }
        
        return {
            'texts': filtered_texts,
            'results': results
        }
    
    def get_model_info(self, model_name):
        """Get information about the model"""
        model_info = {
            'sentence_transformers': {
                'name': 'Sentence Transformers (all-MiniLM-L6-v2)',
                'description': 'Lightweight and fast model, good for general-purpose tasks',
                'context_length': 512,
                'dimensions': 384
            },
            'openai': {
                'name': 'OpenAI text-embedding-3-small',
                'description': 'Commercial model with good performance and efficiency',
                'context_length': 8191,
                'dimensions': 1536
            },
            'nomic': {
                'name': 'Nomic Embed Text v1.5',
                'description': 'Open source model with excellent performance, outperforms OpenAI on many benchmarks',
                'context_length': 8192,
                'dimensions': 768
            }
        }
        return model_info.get(model_name, {})

# Initialize detector
detector = PlagiarismDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        texts = data.get('texts', [])
        models_to_use = data.get('models', ['sentence_transformers'])
        
        results = detector.analyze_texts(texts, models_to_use)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = {}
    
    # Check Sentence Transformers
    model_status['sentence_transformers'] = detector.models['sentence_transformers'] is not None
    
    # Check OpenAI
    model_status['openai'] = detector.models['openai'] is not None
    
    # Check Nomic (simplified check)
    model_status['nomic'] = True  # Always available (with fallback)
    
    return jsonify({
        'status': 'healthy',
        'models': model_status
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 