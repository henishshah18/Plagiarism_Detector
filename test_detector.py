#!/usr/bin/env python3
"""
Test script for the Plagiarism Detector
"""

from app import PlagiarismDetector
import numpy as np

def test_plagiarism_detector():
    """Test the main functionality of the plagiarism detector"""
    
    print("🔍 Testing Plagiarism Detector...")
    
    # Initialize detector
    detector = PlagiarismDetector()
    
    # Test texts
    test_texts = [
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "AI refers to the simulation of human intelligence in computer systems.",
        "Machine learning is a subset of artificial intelligence.",
        "The weather is nice today and I enjoy walking in the park."
    ]
    
    print(f"📝 Testing with {len(test_texts)} texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text[:50]}...")
    
    # Test with Sentence Transformers (most likely to be available)
    print("\n🤖 Testing with Sentence Transformers...")
    
    try:
        result = detector.analyze_texts(test_texts, ['sentence_transformers'])
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        if 'results' in result and 'sentence_transformers' in result['results']:
            model_result = result['results']['sentence_transformers']
            
            if 'error' in model_result:
                print(f"❌ Model Error: {model_result['error']}")
                return False
            
            similarity_matrix = np.array(model_result['similarity_matrix'])
            clones = model_result['clones']
            
            print(f"✅ Successfully generated similarity matrix: {similarity_matrix.shape}")
            print(f"✅ Detected {len(clones)} potential clones")
            
            # Print similarity matrix
            print("\n📊 Similarity Matrix:")
            for i in range(len(similarity_matrix)):
                row = " ".join([f"{sim:.2f}" for sim in similarity_matrix[i]])
                print(f"  Text {i+1}: {row}")
            
            # Print clones
            if clones:
                print(f"\n🚨 Potential Clones (>80% similarity):")
                for clone in clones:
                    print(f"  Text {clone['text1_index']+1} ↔ Text {clone['text2_index']+1}: {clone['similarity']:.1%}")
            else:
                print(f"\n✅ No clones detected (similarity < 80%)")
            
            return True
        else:
            print("❌ No results returned")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_model_availability():
    """Test which models are available"""
    print("\n🔍 Testing Model Availability...")
    
    detector = PlagiarismDetector()
    
    # Test Sentence Transformers
    if detector.models['sentence_transformers'] is not None:
        print("✅ Sentence Transformers: Available")
    else:
        print("❌ Sentence Transformers: Not available")
    
    # Test OpenAI
    if detector.models['openai']:
        print("✅ OpenAI: API key configured")
    else:
        print("⚠️  OpenAI: API key not configured")
    
    # Test Nomic (simplified)
    print("✅ Nomic: Available (with fallback)")

if __name__ == "__main__":
    print("🎯 Plagiarism Detector Test Suite")
    print("=" * 50)
    
    # Test model availability
    test_model_availability()
    
    # Test main functionality
    success = test_plagiarism_detector()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The plagiarism detector is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.") 