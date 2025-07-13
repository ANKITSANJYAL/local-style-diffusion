#!/usr/bin/env python3
"""
User study generator for LPA evaluation.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datetime import datetime

class UserStudyGenerator:
    """
    Generate user study materials for LPA evaluation.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize user study generator.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.results = {}
        self.prompts_data = {}
        
    def load_experiment_data(self):
        """Load experiment results and prompts data."""
        # Load final results
        final_results_file = self.experiment_dir / "final_results.json"
        if final_results_file.exists():
            with open(final_results_file, 'r') as f:
                results_list = json.load(f)
                # Convert list to dictionary with prompt_id as key
                self.results = {result["prompt_id"]: result for result in results_list}
        
        # Load prompts data
        prompts_file = Path("data/prompts/test_prompts.json")
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                prompts_data = json.load(f)
                self.prompts_data = {p["id"]: p for p in prompts_data.get("prompts", [])}
    
    def generate_comparison_survey(self, num_questions: int = 20) -> Dict[str, Any]:
        """
        Generate A/B comparison survey.
        
        Args:
            num_questions: Number of comparison questions to generate
            
        Returns:
            Survey data structure
        """
        print(f"📋 Generating A/B comparison survey with {num_questions} questions...")
        
        # Get available prompt IDs
        available_prompts = list(self.results.keys())
        
        if len(available_prompts) < num_questions:
            print(f"Warning: Only {len(available_prompts)} prompts available, using all")
            num_questions = len(available_prompts)
        
        # Randomly select prompts
        selected_prompts = random.sample(available_prompts, num_questions)
        
        survey = {
            "survey_id": f"lpa_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": "LPA vs Baseline Image Generation Comparison",
            "description": "Please compare the quality of two generated images for each prompt. Rate which image better matches the prompt and style requirements.",
            "instructions": [
                "For each question, you will see a prompt and two generated images (A and B).",
                "Rate which image better matches the prompt and style requirements.",
                "Consider both content accuracy and style consistency.",
                "You can also provide optional comments for each comparison."
            ],
            "questions": [],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_questions": num_questions,
                "experiment_dir": str(self.experiment_dir)
            }
        }
        
        for i, prompt_id in enumerate(selected_prompts, 1):
            prompt_data = self.prompts_data.get(prompt_id, {})
            result_data = self.results.get(prompt_id, {})
            
            # Get image paths
            lpa_image = result_data.get("lpa", {}).get("image_path", "")
            baseline_image = result_data.get("baseline", {}).get("image_path", "")
            
            if lpa_image and baseline_image:
                question = {
                    "question_id": f"q_{i:03d}",
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_data.get("prompt", ""),
                    "prompt_objects": prompt_data.get("objects", []),
                    "prompt_style": prompt_data.get("style", []),
                    "image_a": {
                        "path": baseline_image,
                        "method": "Baseline (SD v1.5)",
                        "description": "Generated using standard Stable Diffusion v1.5"
                    },
                    "image_b": {
                        "path": lpa_image,
                        "method": "LPA (Local Prompt Adaptation)",
                        "description": "Generated using Local Prompt Adaptation method"
                    },
                    "metrics": {
                        "lpa_style_consistency": result_data.get("lpa", {}).get("evaluation_metrics", {}).get("lpa", {}).get("style_consistency", 0),
                        "lpa_clip_score": result_data.get("lpa", {}).get("evaluation_metrics", {}).get("lpa", {}).get("clip_score", 0),
                        "baseline_clip_score": result_data.get("baseline", {}).get("evaluation_metrics", {}).get("baseline", {}).get("clip_score", 0)
                    }
                }
                
                survey["questions"].append(question)
        
        return survey
    
    def generate_style_consistency_survey(self, num_questions: int = 15) -> Dict[str, Any]:
        """
        Generate style consistency evaluation survey.
        
        Args:
            num_questions: Number of questions to generate
            
        Returns:
            Survey data structure
        """
        print(f"🎨 Generating style consistency survey with {num_questions} questions...")
        
        # Filter prompts with style information
        style_prompts = []
        for prompt_id, prompt_data in self.prompts_data.items():
            if prompt_id in self.results and prompt_data.get("style"):
                style_prompts.append(prompt_id)
        
        if len(style_prompts) < num_questions:
            print(f"Warning: Only {len(style_prompts)} style prompts available, using all")
            num_questions = len(style_prompts)
        
        # Randomly select style prompts
        selected_prompts = random.sample(style_prompts, num_questions)
        
        survey = {
            "survey_id": f"style_consistency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": "Style Consistency Evaluation",
            "description": "Rate how well the generated images match the specified artistic style.",
            "instructions": [
                "For each question, you will see a prompt with a specific artistic style.",
                "Rate how well the generated image matches that style on a scale of 1-5.",
                "Consider color palette, brush strokes, artistic technique, and overall aesthetic.",
                "1 = Poor style match, 5 = Excellent style match"
            ],
            "questions": [],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_questions": num_questions,
                "experiment_dir": str(self.experiment_dir)
            }
        }
        
        for i, prompt_id in enumerate(selected_prompts, 1):
            prompt_data = self.prompts_data.get(prompt_id, {})
            result_data = self.results.get(prompt_id, {})
            
            lpa_image = result_data.get("lpa", {}).get("image_path", "")
            
            if lpa_image:
                question = {
                    "question_id": f"style_q_{i:03d}",
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_data.get("prompt", ""),
                    "target_style": prompt_data.get("style", []),
                    "image_path": lpa_image,
                    "method": "LPA (Local Prompt Adaptation)",
                    "rating_scale": {
                        "1": "Poor style match",
                        "2": "Below average style match", 
                        "3": "Average style match",
                        "4": "Good style match",
                        "5": "Excellent style match"
                    }
                }
                
                survey["questions"].append(question)
        
        return survey
    
    def generate_content_fidelity_survey(self, num_questions: int = 15) -> Dict[str, Any]:
        """
        Generate content fidelity evaluation survey.
        
        Args:
            num_questions: Number of questions to generate
            
        Returns:
            Survey data structure
        """
        print(f"📝 Generating content fidelity survey with {num_questions} questions...")
        
        # Filter prompts with multiple objects
        multi_object_prompts = []
        for prompt_id, prompt_data in self.prompts_data.items():
            if prompt_id in self.results and len(prompt_data.get("objects", [])) > 1:
                multi_object_prompts.append(prompt_id)
        
        if len(multi_object_prompts) < num_questions:
            print(f"Warning: Only {len(multi_object_prompts)} multi-object prompts available, using all")
            num_questions = len(multi_object_prompts)
        
        # Randomly select multi-object prompts
        selected_prompts = random.sample(multi_object_prompts, num_questions)
        
        survey = {
            "survey_id": f"content_fidelity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": "Content Fidelity Evaluation",
            "description": "Rate how well the generated images include all requested objects and elements.",
            "instructions": [
                "For each question, you will see a prompt with multiple objects/elements.",
                "Rate how well the generated image includes all requested objects on a scale of 1-5.",
                "Consider object presence, positioning, and overall scene composition.",
                "1 = Missing most objects, 5 = All objects present and well-positioned"
            ],
            "questions": [],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_questions": num_questions,
                "experiment_dir": str(self.experiment_dir)
            }
        }
        
        for i, prompt_id in enumerate(selected_prompts, 1):
            prompt_data = self.prompts_data.get(prompt_id, {})
            result_data = self.results.get(prompt_id, {})
            
            lpa_image = result_data.get("lpa", {}).get("image_path", "")
            
            if lpa_image:
                question = {
                    "question_id": f"content_q_{i:03d}",
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_data.get("prompt", ""),
                    "required_objects": prompt_data.get("objects", []),
                    "image_path": lpa_image,
                    "method": "LPA (Local Prompt Adaptation)",
                    "rating_scale": {
                        "1": "Missing most objects",
                        "2": "Missing several objects",
                        "3": "Some objects missing or unclear",
                        "4": "Most objects present",
                        "5": "All objects present and well-positioned"
                    }
                }
                
                survey["questions"].append(question)
        
        return survey
    
    def generate_html_survey(self, survey_data: Dict[str, Any], survey_type: str) -> str:
        """
        Generate HTML survey for web-based evaluation.
        
        Args:
            survey_data: Survey data structure
            survey_type: Type of survey (comparison, style, content)
            
        Returns:
            HTML string
        """
        print(f"🌐 Generating HTML survey for {survey_type}...")
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{survey_data['title']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .question {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .prompt {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }}
        .images {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .image-container {{
            flex: 1;
            min-width: 300px;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .rating {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
        }}
        .rating input[type="radio"] {{
            display: none;
        }}
        .rating label {{
            padding: 10px 15px;
            background: #e9ecef;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .rating input[type="radio"]:checked + label {{
            background: #667eea;
            color: white;
        }}
        .comparison {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }}
        .option {{
            flex: 1;
            text-align: center;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .option:hover {{
            border-color: #667eea;
        }}
        .option.selected {{
            border-color: #667eea;
            background: #f8f9ff;
        }}
        .comments {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .submit-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 20px;
        }}
        .submit-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .progress {{
            background: #e9ecef;
            height: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .progress-bar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{survey_data['title']}</h1>
        <p>{survey_data['description']}</p>
        <div class="progress">
            <div class="progress-bar" id="progressBar" style="width: 0%"></div>
        </div>
    </div>
    
    <form id="surveyForm">
"""
        
        # Add questions
        for i, question in enumerate(survey_data['questions']):
            html_template += f"""
        <div class="question" id="question_{i}">
            <h3>Question {i+1}</h3>
            <div class="prompt">
                <strong>Prompt:</strong> {question['prompt_text']}
"""
            
            if 'required_objects' in question:
                html_template += f"""
                <br><strong>Required Objects:</strong> {', '.join(question['required_objects'])}
"""
            
            if 'target_style' in question:
                html_template += f"""
                <br><strong>Target Style:</strong> {', '.join(question['target_style'])}
"""
            
            html_template += """
            </div>
"""
            
            # Add images
            if survey_type == "comparison":
                html_template += f"""
            <div class="images">
                <div class="image-container">
                    <img src="{question['image_a']['path']}" alt="Image A">
                    <p><strong>Image A:</strong> {question['image_a']['method']}</p>
                </div>
                <div class="image-container">
                    <img src="{question['image_b']['path']}" alt="Image B">
                    <p><strong>Image B:</strong> {question['image_b']['method']}</p>
                </div>
            </div>
            <div class="comparison">
                <div class="option" onclick="selectOption(this, 'A')">
                    <h4>Image A is Better</h4>
                    <p>Baseline method produces better results</p>
                </div>
                <div class="option" onclick="selectOption(this, 'B')">
                    <h4>Image B is Better</h4>
                    <p>LPA method produces better results</p>
                </div>
            </div>
            <input type="hidden" name="q_{i}_choice" id="q_{i}_choice">
"""
            else:
                html_template += f"""
            <div class="images">
                <div class="image-container">
                    <img src="{question['image_path']}" alt="Generated Image">
                    <p><strong>Generated Image:</strong> {question['method']}</p>
                </div>
            </div>
            <div class="rating">
"""
                for rating, description in question['rating_scale'].items():
                    html_template += f"""
                <input type="radio" name="q_{i}_rating" id="q_{i}_rating_{rating}" value="{rating}">
                <label for="q_{i}_rating_{rating}">{rating}<br><small>{description}</small></label>
"""
                html_template += """
            </div>
"""
            
            html_template += f"""
            <textarea class="comments" name="q_{i}_comments" placeholder="Optional comments..."></textarea>
        </div>
"""
        
        html_template += """
        <button type="submit" class="submit-btn">Submit Survey</button>
    </form>
    
    <script>
        function selectOption(element, choice) {
            // Remove selection from other options
            const options = element.parentElement.querySelectorAll('.option');
            options.forEach(opt => opt.classList.remove('selected'));
            
            // Select this option
            element.classList.add('selected');
            
            // Set hidden input
            const questionIndex = element.closest('.question').id.split('_')[1];
            document.getElementById('q_' + questionIndex + '_choice').value = choice;
            
            updateProgress();
        }
        
        function updateProgress() {
            const questions = document.querySelectorAll('.question');
            const answered = document.querySelectorAll('input[type="radio"]:checked, input[type="hidden"][value]').length;
            const progress = (answered / questions.length) * 100;
            document.getElementById('progressBar').style.width = progress + '%';
        }
        
        document.getElementById('surveyForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Collect form data
            const formData = new FormData(this);
            const results = {};
            
            for (let [key, value] of formData.entries()) {
                results[key] = value;
            }
            
            // Save results
            const surveyResults = {
                survey_id: '""" + survey_data['survey_id'] + """',
                timestamp: new Date().toISOString(),
                results: results
            };
            
            // Download results as JSON
            const dataStr = JSON.stringify(surveyResults, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'survey_results.json';
            link.click();
            
            alert('Survey completed! Results downloaded.');
        });
        
        // Update progress on page load
        updateProgress();
    </script>
</body>
</html>
"""
        
        return html_template
    
    def save_surveys(self, surveys: Dict[str, Any]):
        """Save all generated surveys."""
        print("💾 Saving surveys...")
        
        # Create surveys directory
        surveys_dir = self.experiment_dir / "user_studies"
        surveys_dir.mkdir(exist_ok=True)
        
        # Save survey data
        for survey_type, survey_data in surveys.items():
            # Save JSON data
            json_file = surveys_dir / f"{survey_type}_survey.json"
            with open(json_file, 'w') as f:
                json.dump(survey_data, f, indent=2)
            
            # Generate and save HTML
            html_content = self.generate_html_survey(survey_data, survey_type)
            html_file = surveys_dir / f"{survey_type}_survey.html"
            with open(html_file, 'w') as f:
                f.write(html_content)
        
        print(f"✅ Surveys saved to: {surveys_dir}")
        
        # Create summary file
        summary = {
            "generated_at": datetime.now().isoformat(),
            "surveys": list(surveys.keys()),
            "files": {
                survey_type: {
                    "json": f"{survey_type}_survey.json",
                    "html": f"{survey_type}_survey.html"
                } for survey_type in surveys.keys()
            },
            "instructions": {
                "comparison": "A/B comparison between LPA and baseline methods",
                "style_consistency": "Style consistency evaluation for LPA images",
                "content_fidelity": "Content fidelity evaluation for multi-object prompts"
            }
        }
        
        summary_file = surveys_dir / "survey_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return surveys_dir
    
    def run_user_study_generation(self, num_comparison: int = 20, num_style: int = 15, num_content: int = 15):
        """Run complete user study generation."""
        print("🚀 Starting user study generation...")
        
        # Load experiment data
        self.load_experiment_data()
        
        # Generate surveys
        surveys = {
            "comparison": self.generate_comparison_survey(num_comparison),
            "style_consistency": self.generate_style_consistency_survey(num_style),
            "content_fidelity": self.generate_content_fidelity_survey(num_content)
        }
        
        # Save surveys
        surveys_dir = self.save_surveys(surveys)
        
        print(f"✅ User study generation complete!")
        print(f"Generated {len(surveys)} surveys:")
        for survey_type, survey_data in surveys.items():
            print(f"  • {survey_type}: {len(survey_data['questions'])} questions")
        print(f"All files saved to: {surveys_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate user study materials for LPA evaluation")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--num-comparison", type=int, default=20, help="Number of comparison questions")
    parser.add_argument("--num-style", type=int, default=15, help="Number of style consistency questions")
    parser.add_argument("--num-content", type=int, default=15, help="Number of content fidelity questions")
    
    args = parser.parse_args()
    
    generator = UserStudyGenerator(args.experiment_dir)
    generator.run_user_study_generation(args.num_comparison, args.num_style, args.num_content)

if __name__ == "__main__":
    main() 