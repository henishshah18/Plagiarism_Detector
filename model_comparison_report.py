#!/usr/bin/env python3
"""
Comprehensive Comparison Report Generator for Embedding Models
Analyzes performance, accuracy, and characteristics of different embedding models
"""

import time
import numpy as np
import pandas as pd
from app import PlagiarismDetector
import json
from datetime import datetime

class ModelComparisonReport:
    def __init__(self):
        self.detector = PlagiarismDetector()
        self.test_cases = self.generate_test_cases()
        self.results = {}
        
    def generate_test_cases(self):
        """Generate comprehensive test cases for model comparison"""
        return {
            'identical_texts': {
                'description': 'Identical texts (should be 100% similar)',
                'texts': [
                    "The quick brown fox jumps over the lazy dog.",
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is a subset of artificial intelligence."
                ],
                'expected_clones': 1  # First two should be identical
            },
            'paraphrased_texts': {
                'description': 'Paraphrased content (should be highly similar)',
                'texts': [
                    "Artificial intelligence is the simulation of human intelligence in machines.",
                    "AI refers to the simulation of human intelligence in computer systems.",
                    "Machine learning uses algorithms to analyze data and make predictions.",
                    "ML employs computational methods to examine information and forecast outcomes."
                ],
                'expected_clones': 2  # Two pairs of similar texts
            },
            'different_topics': {
                'description': 'Different topics (should be low similarity)',
                'texts': [
                    "The weather is sunny today with clear blue skies.",
                    "Quantum computing uses quantum mechanical phenomena for computation.",
                    "The recipe calls for two cups of flour and one egg.",
                    "Basketball is a popular sport played with two teams."
                ],
                'expected_clones': 0  # No similar texts
            },
            'technical_content': {
                'description': 'Technical/scientific content',
                'texts': [
                    "Neural networks are computational models inspired by biological neurons.",
                    "Deep learning networks are computational architectures modeled after brain neurons.",
                    "Photosynthesis is the process by which plants convert sunlight into energy.",
                    "The mitochondria is the powerhouse of the cell."
                ],
                'expected_clones': 1  # First two should be similar
            },
            'short_vs_long': {
                'description': 'Mixed length texts',
                'texts': [
                    "AI is smart.",
                    "Artificial intelligence represents the development of computer systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, problem-solving, and pattern recognition.",
                    "Cars are fast.",
                    "Automotive vehicles are designed for rapid transportation."
                ],
                'expected_clones': 0  # Different topics despite length variation
            }
        }
    
    def test_model_performance(self, model_name):
        """Test a specific model's performance across all test cases"""
        print(f"\n🔍 Testing {model_name}...")
        
        model_results = {
            'model_name': model_name,
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'average_processing_time': 0,
            'accuracy_score': 0,
            'test_case_results': {},
            'performance_metrics': {}
        }
        
        total_time = 0
        accuracy_scores = []
        
        for test_name, test_case in self.test_cases.items():
            print(f"  📝 Testing: {test_case['description']}")
            
            try:
                start_time = time.time()
                
                # Run the analysis
                result = self.detector.analyze_texts(test_case['texts'], [model_name])
                
                end_time = time.time()
                processing_time = end_time - start_time
                total_time += processing_time
                
                if 'error' in result:
                    print(f"    ❌ Error: {result['error']}")
                    model_results['failed_tests'] += 1
                    continue
                
                if model_name not in result['results']:
                    print(f"    ❌ Model {model_name} not found in results")
                    model_results['failed_tests'] += 1
                    continue
                
                model_result = result['results'][model_name]
                
                if 'error' in model_result:
                    print(f"    ❌ Model error: {model_result['error']}")
                    model_results['failed_tests'] += 1
                    continue
                
                # Analyze results
                similarity_matrix = np.array(model_result['similarity_matrix'])
                clones = model_result['clones']
                detected_clones = len(clones)
                expected_clones = test_case['expected_clones']
                
                # Calculate accuracy (how close we are to expected number of clones)
                if expected_clones == 0:
                    accuracy = 1.0 if detected_clones == 0 else 0.0
                else:
                    accuracy = 1.0 - abs(detected_clones - expected_clones) / max(expected_clones, detected_clones)
                
                accuracy_scores.append(accuracy)
                
                # Store detailed results
                model_results['test_case_results'][test_name] = {
                    'processing_time': processing_time,
                    'similarity_matrix': similarity_matrix.tolist(),
                    'detected_clones': detected_clones,
                    'expected_clones': expected_clones,
                    'accuracy': accuracy,
                    'clone_details': clones
                }
                
                print(f"    ✅ Processed in {processing_time:.2f}s")
                print(f"    📊 Detected {detected_clones} clones (expected {expected_clones})")
                print(f"    🎯 Accuracy: {accuracy:.1%}")
                
                model_results['successful_tests'] += 1
                
            except Exception as e:
                print(f"    ❌ Exception: {e}")
                model_results['failed_tests'] += 1
            
            model_results['total_tests'] += 1
        
        # Calculate overall metrics
        if model_results['successful_tests'] > 0:
            model_results['average_processing_time'] = total_time / model_results['successful_tests']
            model_results['accuracy_score'] = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Performance metrics
        model_results['performance_metrics'] = {
            'success_rate': model_results['successful_tests'] / model_results['total_tests'],
            'average_processing_time': model_results['average_processing_time'],
            'accuracy_score': model_results['accuracy_score']
        }
        
        return model_results
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("🎯 Generating Comprehensive Model Comparison Report")
        print("=" * 80)
        
        # Test all available models
        available_models = ['sentence_transformers']
        
        # Check if other models are available
        if self.detector.models['openai']:
            available_models.append('openai')
        if self.detector.models['nomic']:
            available_models.append('nomic')
        
        print(f"📊 Testing {len(available_models)} models: {', '.join(available_models)}")
        
        # Test each model
        for model_name in available_models:
            model_results = self.test_model_performance(model_name)
            self.results[model_name] = model_results
        
        # Generate detailed comparison
        self.print_detailed_comparison()
        
        # Save results to file
        self.save_report_to_file()
        
        return self.results
    
    def print_detailed_comparison(self):
        """Print detailed comparison analysis"""
        print("\n" + "=" * 80)
        print("📊 DETAILED MODEL COMPARISON REPORT")
        print("=" * 80)
        
        # Summary table
        print("\n📈 PERFORMANCE SUMMARY")
        print("-" * 60)
        
        summary_data = []
        for model_name, results in self.results.items():
            model_info = self.detector.get_model_info(model_name)
            summary_data.append({
                'Model': model_info.get('name', model_name),
                'Success Rate': f"{results['performance_metrics']['success_rate']:.1%}",
                'Avg Time (s)': f"{results['performance_metrics']['average_processing_time']:.2f}",
                'Accuracy': f"{results['performance_metrics']['accuracy_score']:.1%}",
                'Context Length': model_info.get('context_length', 'N/A'),
                'Dimensions': model_info.get('dimensions', 'N/A')
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Detailed analysis
        print("\n🔍 DETAILED ANALYSIS")
        print("-" * 60)
        
        for model_name, results in self.results.items():
            model_info = self.detector.get_model_info(model_name)
            print(f"\n🤖 {model_info.get('name', model_name)}")
            print(f"   Description: {model_info.get('description', 'N/A')}")
            print(f"   Overall Accuracy: {results['performance_metrics']['accuracy_score']:.1%}")
            print(f"   Average Processing Time: {results['performance_metrics']['average_processing_time']:.2f}s")
            print(f"   Success Rate: {results['performance_metrics']['success_rate']:.1%}")
            
            # Test case breakdown
            print("   📝 Test Case Performance:")
            for test_name, test_result in results['test_case_results'].items():
                test_desc = self.test_cases[test_name]['description']
                print(f"     • {test_desc}")
                print(f"       Accuracy: {test_result['accuracy']:.1%}, Time: {test_result['processing_time']:.2f}s")
        
        # Recommendations
        self.print_recommendations()
    
    def print_recommendations(self):
        """Print model recommendations based on analysis"""
        print("\n💡 RECOMMENDATIONS")
        print("-" * 60)
        
        if not self.results:
            print("No models were successfully tested.")
            return
        
        # Find best performing model
        best_accuracy = max(r['performance_metrics']['accuracy_score'] for r in self.results.values())
        best_speed = min(r['performance_metrics']['average_processing_time'] for r in self.results.values() if r['performance_metrics']['average_processing_time'] > 0)
        
        print("🏆 BEST MODELS BY CATEGORY:")
        
        for model_name, results in self.results.items():
            model_info = self.detector.get_model_info(model_name)
            metrics = results['performance_metrics']
            
            recommendations = []
            
            if metrics['accuracy_score'] == best_accuracy:
                recommendations.append("🎯 HIGHEST ACCURACY")
            
            if metrics['average_processing_time'] == best_speed:
                recommendations.append("⚡ FASTEST PROCESSING")
            
            if metrics['success_rate'] == 1.0:
                recommendations.append("✅ MOST RELIABLE")
            
            if recommendations:
                print(f"\n{model_info.get('name', model_name)}: {' | '.join(recommendations)}")
        
        print("\n📋 USE CASE RECOMMENDATIONS:")
        print("• For Production Applications: Choose the model with highest accuracy and reliability")
        print("• For Real-time Processing: Choose the fastest model")
        print("• For Research: Compare multiple models to understand different perspectives")
        print("• For Cost-sensitive Applications: Use open-source models (Sentence Transformers, Nomic)")
        print("• For Maximum Accuracy: Consider using commercial models (OpenAI) with API keys")
    
    def save_report_to_file(self):
        """Save detailed report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_report_{timestamp}.json"
        
        # Prepare data for JSON serialization
        report_data = {
            'timestamp': timestamp,
            'test_cases': self.test_cases,
            'results': self.results,
            'summary': {
                'total_models_tested': len(self.results),
                'total_test_cases': len(self.test_cases)
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving report: {e}")

def main():
    """Main function to run the comparison report"""
    print("🚀 Starting Comprehensive Model Comparison Analysis")
    print("=" * 80)
    
    # Create and run the comparison
    comparison = ModelComparisonReport()
    results = comparison.generate_comparison_report()
    
    print("\n" + "=" * 80)
    print("🎉 MODEL COMPARISON REPORT COMPLETED")
    print("=" * 80)
    
    print("\n📊 QUICK SUMMARY:")
    for model_name, result in results.items():
        model_info = comparison.detector.get_model_info(model_name)
        print(f"• {model_info.get('name', model_name)}: "
              f"{result['performance_metrics']['accuracy_score']:.1%} accuracy, "
              f"{result['performance_metrics']['average_processing_time']:.2f}s avg time")
    
    print("\n📁 Check the generated JSON file for detailed results and analysis.")

if __name__ == "__main__":
    main() 