"""
Prompt parsing utilities for Local Prompt Adaptation (LPA).

This module provides functionality to parse text prompts and separate them into
object tokens and style tokens for controlled cross-attention injection.
"""

import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Using rule-based parsing.")


@dataclass
class ParsedPrompt:
    """Data class for storing parsed prompt information."""
    original_prompt: str
    object_tokens: List[str]
    style_tokens: List[str]
    confidence_scores: Dict[str, float]
    prompt_hash: str
    complexity: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_prompt": self.original_prompt,
            "object_tokens": self.object_tokens,
            "style_tokens": self.style_tokens,
            "confidence_scores": self.confidence_scores,
            "prompt_hash": self.prompt_hash,
            "complexity": self.complexity
        }


class PromptParser:
    """
    Parser for separating prompts into object and style tokens.
    
    Supports both spaCy-based parsing and rule-based fallback.
    """
    
    def __init__(self, method: str = "spacy", spacy_model: str = "en_core_web_sm"):
        """
        Initialize the prompt parser.
        
        Args:
            method: Parsing method ("spacy" or "rule_based")
            spacy_model: spaCy model to use for parsing
        """
        self.method = method
        self.spacy_model = spacy_model
        self.nlp = None
        
        # Style keywords for rule-based parsing
        self.style_keywords = {
            "artistic": ["style", "art", "painting", "drawing", "illustration"],
            "photographic": ["photo", "photograph", "realistic", "photorealistic"],
            "digital": ["digital", "3d", "rendered", "computer", "cg"],
            "traditional": ["oil", "watercolor", "acrylic", "pastel", "charcoal"],
            "modern": ["modern", "contemporary", "abstract", "minimalist"],
            "vintage": ["vintage", "retro", "classic", "old", "antique"],
            "fantasy": ["fantasy", "magical", "mythical", "enchanted"],
            "sci_fi": ["sci-fi", "cyberpunk", "futuristic", "space", "alien"],
            "anime": ["anime", "manga", "cartoon", "cel-shaded"],
            "realistic": ["realistic", "photorealistic", "hyperrealistic"]
        }
        
        # Prepositions that indicate spatial relationships
        self.spatial_prepositions = [
            "on", "in", "at", "next to", "beside", "behind", "in front of",
            "above", "below", "under", "over", "inside", "outside", "between",
            "among", "around", "through", "across", "along", "against"
        ]
        
        if method == "spacy" and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                print(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                print(f"spaCy model {spacy_model} not found. Using rule-based parsing.")
                self.method = "rule_based"
        else:
            self.method = "rule_based"
            print("Using rule-based prompt parsing.")
    
    def parse_prompt(self, prompt: str) -> ParsedPrompt:
        """
        Parse a prompt into object and style tokens.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            ParsedPrompt object with separated tokens
        """
        prompt = prompt.strip()
        prompt_hash = self._generate_prompt_hash(prompt)
        
        if self.method == "spacy" and self.nlp is not None:
            return self._parse_with_spacy(prompt, prompt_hash)
        else:
            return self._parse_rule_based(prompt, prompt_hash)
    
    def _parse_with_spacy(self, prompt: str, prompt_hash: str) -> ParsedPrompt:
        """Parse prompt using spaCy NLP."""
        doc = self.nlp(prompt.lower())
        
        object_tokens = []
        style_tokens = []
        confidence_scores = {}
        
        # Extract nouns and noun phrases as potential objects
        for chunk in doc.noun_chunks:
            # Skip style-related nouns
            if not self._is_style_token(chunk.text):
                object_tokens.append(chunk.text)
                confidence_scores[chunk.text] = 0.8
        
        # Extract individual nouns
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN"] and 
                token.text not in object_tokens and 
                not self._is_style_token(token.text)):
                object_tokens.append(token.text)
                confidence_scores[token.text] = 0.7
        
        # Extract style tokens
        for token in doc:
            if self._is_style_token(token.text):
                style_tokens.append(token.text)
                confidence_scores[token.text] = 0.9
        
        # Look for style phrases
        style_phrases = self._extract_style_phrases(doc)
        style_tokens.extend(style_phrases)
        
        # Remove duplicates while preserving order
        object_tokens = list(dict.fromkeys(object_tokens))
        style_tokens = list(dict.fromkeys(style_tokens))
        
        complexity = self._assess_complexity(doc, len(object_tokens), len(style_tokens))
        
        return ParsedPrompt(
            original_prompt=prompt,
            object_tokens=object_tokens,
            style_tokens=style_tokens,
            confidence_scores=confidence_scores,
            prompt_hash=prompt_hash,
            complexity=complexity
        )
    
    def _parse_rule_based(self, prompt: str, prompt_hash: str) -> ParsedPrompt:
        """Parse prompt using rule-based approach."""
        prompt_lower = prompt.lower()
        
        object_tokens = []
        style_tokens = []
        confidence_scores = {}
        
        # Extract style tokens first
        for style_category, keywords in self.style_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    style_tokens.append(keyword)
                    confidence_scores[keyword] = 0.8
        
        # Look for style phrases (e.g., "in X style", "X style")
        style_patterns = [
            r'in (\w+(?:\s+\w+)*) style',
            r'(\w+(?:\s+\w+)*) style',
            r'(\w+(?:\s+\w+)*) art',
            r'(\w+(?:\s+\w+)*) painting'
        ]
        
        for pattern in style_patterns:
            matches = re.findall(pattern, prompt_lower)
            for match in matches:
                if match not in style_tokens:
                    style_tokens.append(match)
                    confidence_scores[match] = 0.9
        
        # Extract potential objects (nouns)
        # Simple heuristic: words that are not style-related and not common stop words
        stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        words = re.findall(r'\b\w+\b', prompt_lower)
        for word in words:
            if (word not in stop_words and 
                word not in style_tokens and 
                len(word) > 2 and
                not self._is_style_token(word)):
                object_tokens.append(word)
                confidence_scores[word] = 0.6
        
        # Remove duplicates
        object_tokens = list(dict.fromkeys(object_tokens))
        style_tokens = list(dict.fromkeys(style_tokens))
        
        complexity = self._assess_complexity_rule_based(len(object_tokens), len(style_tokens))
        
        return ParsedPrompt(
            original_prompt=prompt,
            object_tokens=object_tokens,
            style_tokens=style_tokens,
            confidence_scores=confidence_scores,
            prompt_hash=prompt_hash,
            complexity=complexity
        )
    
    def _is_style_token(self, token: str) -> bool:
        """Check if a token is style-related."""
        token_lower = token.lower()
        
        # Check against style keywords
        for keywords in self.style_keywords.values():
            if token_lower in keywords:
                return True
        
        # Check for style-related patterns
        style_patterns = ["style", "art", "painting", "drawing", "illustration"]
        return any(pattern in token_lower for pattern in style_patterns)
    
    def _extract_style_phrases(self, doc: Doc) -> List[str]:
        """Extract style phrases from spaCy document."""
        style_phrases = []
        
        # Look for "in X style" patterns
        for token in doc:
            if token.text == "in" and token.dep_ == "prep":
                for child in token.rights:
                    if "style" in child.text:
                        phrase = f"in {child.text}"
                        style_phrases.append(phrase)
        
        return style_phrases
    
    def _assess_complexity(self, doc: Doc, num_objects: int, num_styles: int) -> str:
        """Assess prompt complexity using spaCy analysis."""
        # Count different parts of speech
        num_nouns = len([token for token in doc if token.pos_ == "NOUN"])
        num_verbs = len([token for token in doc if token.pos_ == "VERB"])
        num_adj = len([token for token in doc if token.pos_ == "ADJ"])
        
        # Count spatial relationships
        spatial_count = len([token for token in doc if token.text in self.spatial_prepositions])
        
        total_complexity = num_objects + num_styles + spatial_count + (num_adj // 2)
        
        if total_complexity <= 3:
            return "low"
        elif total_complexity <= 6:
            return "medium"
        elif total_complexity <= 9:
            return "high"
        else:
            return "very_high"
    
    def _assess_complexity_rule_based(self, num_objects: int, num_styles: int) -> str:
        """Assess prompt complexity using rule-based approach."""
        total_complexity = num_objects + num_styles
        
        if total_complexity <= 2:
            return "low"
        elif total_complexity <= 4:
            return "medium"
        elif total_complexity <= 6:
            return "high"
        else:
            return "very_high"
    
    def _generate_prompt_hash(self, prompt: str) -> str:
        """Generate a hash for the prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    def load_prompts_from_file(self, file_path: str) -> List[ParsedPrompt]:
        """Load and parse prompts from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        parsed_prompts = []
        for prompt_data in data.get("prompts", []):
            # If the file already contains parsed data, use it
            if "object_tokens" in prompt_data and "style_tokens" in prompt_data:
                parsed_prompt = ParsedPrompt(
                    original_prompt=prompt_data["prompt"],
                    object_tokens=prompt_data["objects"],
                    style_tokens=prompt_data["style"],
                    confidence_scores={token: 0.9 for token in prompt_data["objects"] + prompt_data["style"]},
                    prompt_hash=self._generate_prompt_hash(prompt_data["prompt"]),
                    complexity=prompt_data.get("complexity", "medium")
                )
            else:
                # Parse the prompt
                parsed_prompt = self.parse_prompt(prompt_data["prompt"])
            
            parsed_prompts.append(parsed_prompt)
        
        return parsed_prompts
    
    def save_parsed_prompts(self, parsed_prompts: List[ParsedPrompt], file_path: str):
        """Save parsed prompts to a JSON file."""
        data = {
            "metadata": {
                "description": "Parsed prompts for LPA evaluation",
                "total_prompts": len(parsed_prompts),
                "parser_method": self.method,
                "version": "1.0"
            },
            "parsed_prompts": [prompt.to_dict() for prompt in parsed_prompts]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_statistics(self, parsed_prompts: List[ParsedPrompt]) -> Dict[str, Any]:
        """Get statistics about parsed prompts."""
        total_prompts = len(parsed_prompts)
        
        # Count objects and styles
        total_objects = sum(len(p.object_tokens) for p in parsed_prompts)
        total_styles = sum(len(p.style_tokens) for p in parsed_prompts)
        
        # Complexity distribution
        complexity_counts = {}
        for prompt in parsed_prompts:
            complexity_counts[prompt.complexity] = complexity_counts.get(prompt.complexity, 0) + 1
        
        # Average confidence scores
        avg_confidence = 0
        total_tokens = 0
        for prompt in parsed_prompts:
            for score in prompt.confidence_scores.values():
                avg_confidence += score
                total_tokens += 1
        
        if total_tokens > 0:
            avg_confidence /= total_tokens
        
        return {
            "total_prompts": total_prompts,
            "total_objects": total_objects,
            "total_styles": total_styles,
            "avg_objects_per_prompt": total_objects / total_prompts if total_prompts > 0 else 0,
            "avg_styles_per_prompt": total_styles / total_prompts if total_prompts > 0 else 0,
            "complexity_distribution": complexity_counts,
            "average_confidence": avg_confidence
        } 