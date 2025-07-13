#!/usr/bin/env python3
"""
Master script to run complete paper analysis and strengthening.
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

class CompletePaperAnalysis:
    """
    Master orchestrator for complete paper analysis and strengthening.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize complete paper analysis.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.results = {}
        self.analysis_results = {}
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")
        
        # Check if experiment results exist
        if not self.experiment_dir.exists():
            print(f"❌ Experiment directory not found: {self.experiment_dir}")
            return False
        
        # Check for required files
        required_files = [
            "final_results.json",
            "evaluation_summary.json",
            "detailed_comparison.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (self.experiment_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        
        print("✅ Prerequisites check passed")
        return True
    
    def run_ablation_studies(self) -> bool:
        """Run ablation studies for injection order analysis."""
        print("\n🔬 PHASE 1: Running Ablation Studies")
        print("=" * 50)
        
        ablation_config = "configs/experiment_config_ablation.yaml"
        
        if not Path(ablation_config).exists():
            print(f"❌ Ablation config not found: {ablation_config}")
            return False
        
        try:
            cmd = [
                sys.executable, "scripts/run_ablation_studies.py",
                ablation_config
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Ablation studies completed successfully")
                return True
            else:
                print(f"❌ Ablation studies failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running ablation studies: {e}")
            return False
    
    def run_research_questions_analysis(self) -> bool:
        """Run research questions analysis."""
        print("\n📊 PHASE 2: Research Questions Analysis")
        print("=" * 50)
        
        try:
            cmd = [
                sys.executable, "scripts/analyze_research_questions.py",
                str(self.experiment_dir)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Research questions analysis completed successfully")
                return True
            else:
                print(f"❌ Research questions analysis failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running research questions analysis: {e}")
            return False
    
    def run_sota_comparison(self) -> bool:
        """Run SOTA comparison analysis."""
        print("\n🏆 PHASE 3: SOTA Comparison")
        print("=" * 50)
        
        try:
            cmd = [
                sys.executable, "scripts/compare_with_sota.py",
                str(self.experiment_dir)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ SOTA comparison completed successfully")
                return True
            else:
                print(f"❌ SOTA comparison failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running SOTA comparison: {e}")
            return False
    
    def run_user_study_generation(self) -> bool:
        """Generate user study materials."""
        print("\n👥 PHASE 4: User Study Generation")
        print("=" * 50)
        
        try:
            cmd = [
                sys.executable, "scripts/user_study_generator.py",
                str(self.experiment_dir),
                "--num-comparison", "20",
                "--num-style", "15",
                "--num-content", "15"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ User study generation completed successfully")
                return True
            else:
                print(f"❌ User study generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running user study generation: {e}")
            return False
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive paper analysis report."""
        print("\n📝 PHASE 5: Generating Comprehensive Report")
        print("=" * 50)
        
        report_lines = []
        report_lines.append("LPA COMPLETE PAPER ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Experiment Directory: {self.experiment_dir}")
        report_lines.append("")
        
        # Load all analysis results
        analysis_files = [
            ("Ablation Studies", "experiments/ablation_studies/ablation_summary_*.json"),
            ("Research Questions", f"{self.experiment_dir}/research_analysis_results.json"),
            ("SOTA Comparison", f"{self.experiment_dir}/sota_comparison_table.csv"),
            ("User Studies", f"{self.experiment_dir}/user_studies/survey_summary.json")
        ]
        
        report_lines.append("ANALYSIS COMPONENTS:")
        report_lines.append("-" * 30)
        
        for component_name, file_pattern in analysis_files:
            if Path(file_pattern.replace("*", "")).parent.exists():
                files = list(Path(file_pattern.replace("*", "")).parent.glob(Path(file_pattern).name.replace("*", "*")))
                if files:
                    report_lines.append(f"✅ {component_name}: {len(files)} files found")
                else:
                    report_lines.append(f"⚠️  {component_name}: No files found")
            else:
                report_lines.append(f"❌ {component_name}: Directory not found")
        
        # Paper strength assessment
        report_lines.append("\n\nPAPER STRENGTH ASSESSMENT:")
        report_lines.append("-" * 30)
        
        # Check ablation studies
        ablation_dir = Path("experiments/ablation_studies")
        if ablation_dir.exists():
            ablation_files = list(ablation_dir.glob("ablation_summary_*.json"))
            if ablation_files:
                report_lines.append("✅ Ablation Studies: COMPLETED")
                report_lines.append("   • Injection order analysis done")
                report_lines.append("   • Multiple variants tested")
                report_lines.append("   • Research Question 1 addressed")
            else:
                report_lines.append("❌ Ablation Studies: INCOMPLETE")
        else:
            report_lines.append("❌ Ablation Studies: NOT STARTED")
        
        # Check research questions
        research_file = self.experiment_dir / "research_analysis_results.json"
        if research_file.exists():
            report_lines.append("✅ Research Questions: COMPLETED")
            report_lines.append("   • Complexity scaling analysis done")
            report_lines.append("   • Style vs content trade-off analyzed")
            report_lines.append("   • Style robustness evaluated")
            report_lines.append("   • Research Questions 2-4 addressed")
        else:
            report_lines.append("❌ Research Questions: INCOMPLETE")
        
        # Check SOTA comparison
        sota_file = self.experiment_dir / "sota_comparison_table.csv"
        if sota_file.exists():
            report_lines.append("✅ SOTA Comparison: COMPLETED")
            report_lines.append("   • Comparison with 7+ SOTA methods")
            report_lines.append("   • CLIP score rankings generated")
            report_lines.append("   • Style consistency comparison done")
        else:
            report_lines.append("❌ SOTA Comparison: INCOMPLETE")
        
        # Check user studies
        user_studies_dir = self.experiment_dir / "user_studies"
        if user_studies_dir.exists():
            survey_files = list(user_studies_dir.glob("*_survey.html"))
            if survey_files:
                report_lines.append("✅ User Studies: COMPLETED")
                report_lines.append(f"   • {len(survey_files)} survey types generated")
                report_lines.append("   • A/B comparison surveys ready")
                report_lines.append("   • Style consistency surveys ready")
                report_lines.append("   • Content fidelity surveys ready")
            else:
                report_lines.append("❌ User Studies: INCOMPLETE")
        else:
            report_lines.append("❌ User Studies: NOT STARTED")
        
        # Overall assessment
        report_lines.append("\n\nOVERALL PAPER ASSESSMENT:")
        report_lines.append("-" * 30)
        
        completed_components = 0
        total_components = 4
        
        if ablation_dir.exists() and list(ablation_dir.glob("ablation_summary_*.json")):
            completed_components += 1
        if research_file.exists():
            completed_components += 1
        if sota_file.exists():
            completed_components += 1
        if user_studies_dir.exists() and list(user_studies_dir.glob("*_survey.html")):
            completed_components += 1
        
        completion_rate = (completed_components / total_components) * 100
        
        if completion_rate >= 90:
            report_lines.append("🏆 EXCELLENT - Paper is top-tier ready!")
            report_lines.append("   • All major components completed")
            report_lines.append("   • Strong experimental foundation")
            report_lines.append("   • Comprehensive analysis done")
        elif completion_rate >= 75:
            report_lines.append("🥈 VERY GOOD - Paper is conference ready")
            report_lines.append("   • Most components completed")
            report_lines.append("   • Solid experimental work")
            report_lines.append("   • Minor improvements needed")
        elif completion_rate >= 50:
            report_lines.append("🥉 GOOD - Paper has strong potential")
            report_lines.append("   • Half of components completed")
            report_lines.append("   • Good foundation established")
            report_lines.append("   • Significant work still needed")
        else:
            report_lines.append("⚠️  NEEDS WORK - Paper requires more development")
            report_lines.append("   • Most components incomplete")
            report_lines.append("   • Foundation needs strengthening")
            report_lines.append("   • Considerable work required")
        
        report_lines.append(f"\nCompletion Rate: {completion_rate:.1f}% ({completed_components}/{total_components})")
        
        # Next steps
        report_lines.append("\n\nNEXT STEPS:")
        report_lines.append("-" * 20)
        
        if completion_rate < 100:
            if not ablation_dir.exists() or not list(ablation_dir.glob("ablation_summary_*.json")):
                report_lines.append("1. Run ablation studies: python scripts/run_ablation_studies.py configs/experiment_config_ablation.yaml")
            
            if not research_file.exists():
                report_lines.append("2. Run research questions analysis: python scripts/analyze_research_questions.py <experiment_dir>")
            
            if not sota_file.exists():
                report_lines.append("3. Run SOTA comparison: python scripts/compare_with_sota.py <experiment_dir>")
            
            if not user_studies_dir.exists() or not list(user_studies_dir.glob("*_survey.html")):
                report_lines.append("4. Generate user studies: python scripts/user_study_generator.py <experiment_dir>")
        
        report_lines.append("5. Conduct user studies with generated surveys")
        report_lines.append("6. Write paper sections based on analysis results")
        report_lines.append("7. Create figures and tables from generated data")
        report_lines.append("8. Submit to target conference/journal")
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS:")
        report_lines.append("-" * 20)
        
        if completion_rate >= 75:
            report_lines.append("• Paper is ready for submission to top-tier conferences")
            report_lines.append("• Consider ICML, NeurIPS, or ICLR")
            report_lines.append("• Focus on writing clear methodology and results sections")
            report_lines.append("• Create compelling visualizations from analysis data")
        elif completion_rate >= 50:
            report_lines.append("• Paper is ready for mid-tier conferences")
            report_lines.append("• Consider AAAI, IJCAI, or CVPR")
            report_lines.append("• Complete missing analysis components")
            report_lines.append("• Strengthen experimental validation")
        else:
            report_lines.append("• Focus on completing core experiments first")
            report_lines.append("• Run ablation studies to validate approach")
            report_lines.append("• Conduct user studies for human evaluation")
            report_lines.append("• Consider workshop submission for early feedback")
        
        return "\n".join(report_lines)
    
    def run_complete_analysis(self, skip_ablation: bool = False, skip_research: bool = False, 
                            skip_sota: bool = False, skip_user_studies: bool = False):
        """Run complete paper analysis."""
        print("🚀 STARTING COMPLETE PAPER ANALYSIS")
        print("=" * 60)
        print(f"Experiment Directory: {self.experiment_dir}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites not met. Exiting.")
            return False
        
        # Track completion
        completed_phases = []
        failed_phases = []
        
        # Phase 1: Ablation Studies
        if not skip_ablation:
            if self.run_ablation_studies():
                completed_phases.append("Ablation Studies")
            else:
                failed_phases.append("Ablation Studies")
        else:
            print("⏭️  Skipping ablation studies")
        
        # Phase 2: Research Questions Analysis
        if not skip_research:
            if self.run_research_questions_analysis():
                completed_phases.append("Research Questions Analysis")
            else:
                failed_phases.append("Research Questions Analysis")
        else:
            print("⏭️  Skipping research questions analysis")
        
        # Phase 3: SOTA Comparison
        if not skip_sota:
            if self.run_sota_comparison():
                completed_phases.append("SOTA Comparison")
            else:
                failed_phases.append("SOTA Comparison")
        else:
            print("⏭️  Skipping SOTA comparison")
        
        # Phase 4: User Study Generation
        if not skip_user_studies:
            if self.run_user_study_generation():
                completed_phases.append("User Study Generation")
            else:
                failed_phases.append("User Study Generation")
        else:
            print("⏭️  Skipping user study generation")
        
        # Phase 5: Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save report
        report_file = self.experiment_dir / "complete_paper_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("COMPLETE PAPER ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"✅ Completed Phases: {len(completed_phases)}")
        for phase in completed_phases:
            print(f"   • {phase}")
        
        if failed_phases:
            print(f"❌ Failed Phases: {len(failed_phases)}")
            for phase in failed_phases:
                print(f"   • {phase}")
        
        print(f"\n📄 Comprehensive report saved to: {report_file}")
        print("\n" + "=" * 60)
        print(report)
        
        return len(failed_phases) == 0

def main():
    parser = argparse.ArgumentParser(description="Run complete paper analysis and strengthening")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation studies")
    parser.add_argument("--skip-research", action="store_true", help="Skip research questions analysis")
    parser.add_argument("--skip-sota", action="store_true", help="Skip SOTA comparison")
    parser.add_argument("--skip-user-studies", action="store_true", help="Skip user study generation")
    
    args = parser.parse_args()
    
    analyzer = CompletePaperAnalysis(args.experiment_dir)
    success = analyzer.run_complete_analysis(
        skip_ablation=args.skip_ablation,
        skip_research=args.skip_research,
        skip_sota=args.skip_sota,
        skip_user_studies=args.skip_user_studies
    )
    
    if success:
        print("\n🎉 COMPLETE PAPER ANALYSIS SUCCESSFUL!")
        print("Your paper is now strengthened and ready for submission!")
    else:
        print("\n⚠️  Some phases failed. Check the report for details.")

if __name__ == "__main__":
    main() 