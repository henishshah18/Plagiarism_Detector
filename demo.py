#!/usr/bin/env python3
"""
Demo script showing how to use the Plagiarism Detector
"""

from app import PlagiarismDetector
import json

def demo_plagiarism_detection():
    """Demonstrate plagiarism detection with sample texts"""
    
    print("🎯 Plagiarism Detector Demo")
    print("=" * 60)
    
    # Initialize the detector
    detector = PlagiarismDetector()
    
    # Sample texts for demonstration
    sample_texts = [
        "Artificial intelligence is a branch of computer science that aims to create machines that can perform tasks requiring human intelligence.",
        "AI is a field of computer science focused on building machines capable of performing tasks that typically require human intelligence.",
        "Machine learning is a subset of artificial intelligence that uses statistical techniques to give computers the ability to learn from data.",
        "The weather forecast shows sunny skies with temperatures reaching 75 degrees Fahrenheit today.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
        "Global warming is the gradual increase in Earth's average surface temperature due to human activities and greenhouse gas emissions."
    ]
    
    print(f"📝 Sample Texts ({len(sample_texts)} total):")
    for i, text in enumerate(sample_texts, 1):
        print(f"  {i}. {text}")
    
    print("\n" + "=" * 60)
    
    # Test with available models
    available_models = ['sentence_transformers']
    
    # Add other models if available
    if detector.models['openai']:
        available_models.append('openai')
    if detector.models['nomic']:
        available_models.append('nomic')
    
    print(f"🤖 Testing with available models: {', '.join(available_models)}")
    
    # Analyze texts
    results = detector.analyze_texts(sample_texts, available_models)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n📊 Analysis Results:")
    print("=" * 60)
    
    for model_name, model_results in results['results'].items():
        print(f"\n🔍 {model_results['model_info']['name']}")
        print(f"   {model_results['model_info']['description']}")
        
        if 'error' in model_results:
            print(f"   ❌ Error: {model_results['error']}")
            continue
        
        similarity_matrix = model_results['similarity_matrix']
        clones = model_results['clones']
        
        print(f"   📈 Similarity Matrix ({len(similarity_matrix)}x{len(similarity_matrix)}):")
        for i, row in enumerate(similarity_matrix):
            similarities = [f"{sim:.2f}" for sim in row]
            print(f"     Text {i+1}: {' '.join(similarities)}")
        
        print(f"\n   🚨 Potential Clones (>80% similarity): {len(clones)}")
        if clones:
            for clone in clones:
                text1_idx = clone['text1_index']
                text2_idx = clone['text2_index']
                similarity = clone['similarity']
                
                print(f"     • Text {text1_idx + 1} ↔ Text {text2_idx + 1}: {similarity:.1%}")
                print(f"       Text {text1_idx + 1}: {sample_texts[text1_idx][:80]}...")
                print(f"       Text {text2_idx + 1}: {sample_texts[text2_idx][:80]}...")
        else:
            print("     ✅ No potential clones detected")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\n💡 Key Insights:")
    print("   • Texts 1 & 2 should show high similarity (both about AI)")
    print("   • Texts 5 & 6 should show high similarity (both about climate)")
    print("   • Other texts should show low similarity")
    print("   • Different models may produce slightly different results")

def demo_model_comparison():
    """Compare different embedding models"""
    
    print("\n🔬 Model Comparison Demo")
    print("=" * 60)
    
    detector = PlagiarismDetector()
    
    # Simple test case
    test_texts = [
        "The cat sat on the mat.",
        "A feline was sitting on a rug.",
        "Dogs are loyal pets."
    ]
    
    print("📝 Test Texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    
    models_to_test = ['sentence_transformers']
    
    for model_name in models_to_test:
        print(f"\n🤖 Testing with {model_name}:")
        
        try:
            results = detector.analyze_texts(test_texts, [model_name])
            
            if 'error' in results:
                print(f"   ❌ Error: {results['error']}")
                continue
            
            model_result = results['results'][model_name]
            
            if 'error' in model_result:
                print(f"   ❌ Error: {model_result['error']}")
                continue
            
            similarity_matrix = model_result['similarity_matrix']
            
            print(f"   📊 Similarity between Text 1 & 2: {similarity_matrix[0][1]:.1%}")
            print(f"   📊 Similarity between Text 1 & 3: {similarity_matrix[0][2]:.1%}")
            print(f"   📊 Similarity between Text 2 & 3: {similarity_matrix[1][2]:.1%}")
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    # Run the main demo
    demo_plagiarism_detection()
    
    # Run model comparison
    demo_model_comparison()
    
    print("\n" + "=" * 60)
    print("🌐 Web Interface:")
    print("   The Flask web application is running on http://localhost:5000")
    print("   Open your browser and navigate to this URL to use the web interface.")
    print("\n📚 Documentation:")
    print("   Check README.md for detailed setup and usage instructions.")
    print("   The web interface provides an interactive way to test plagiarism detection.") 