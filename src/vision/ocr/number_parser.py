"""
TactiBird Overlay - Number Parser for OCR Results
"""

import re
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class NumberParser:
    """Parse numbers from OCR text for game stats"""
    
    def __init__(self):
        # Common OCR misreads and corrections
        self.ocr_corrections = {
            'O': '0', 'o': '0', 'Q': '0', 'D': '0',
            'l': '1', 'I': '1', '|': '1', 'i': '1',
            'Z': '2', 'z': '2',
            'S': '5', 's': '5',
            'G': '6', 'g': '6',
            'T': '7', 't': '7',
            'B': '8', 'b': '8'
        }
        
        # Number patterns for different contexts
        self.patterns = {
            'gold': [
                r'(\d+)\s*g',           # "50g", "100 g"
                r'gold[:]\s*(\d+)',     # "Gold: 50"
                r'(\d+)\s*gold',        # "50 gold"
                r'^\s*(\d+)\s*$'        # Just a number
            ],
            'health': [
                r'(\d+)\s*hp',          # "75hp", "100 hp"
                r'health[:]\s*(\d+)',   # "Health: 75"
                r'(\d+)\s*/\s*\d+',     # "75/100"
                r'^\s*(\d+)\s*$'        # Just a number
            ],
            'level': [
                r'level\s*(\d+)',       # "Level 5"
                r'lvl\s*(\d+)',         # "Lvl 5"
                r'lv\s*(\d+)',          # "Lv 5"
                r'^\s*(\d+)\s*$'        # Just a number
            ],
            'stage_round': [
                r'(\d+)[-/]\s*(\d+)',   # "2-1", "2/1"
                r'stage\s*(\d+)\s*round\s*(\d+)',  # "Stage 2 Round 1"
                r'(\d+)\s*-\s*(\d+)'    # "2 - 1"
            ],
            'experience': [
                r'(\d+)\s*/\s*(\d+)',   # "4/6"
                r'exp[:]\s*(\d+)',      # "Exp: 4"
                r'xp[:]\s*(\d+)',       # "XP: 4"
                r'^\s*(\d+)\s*$'        # Just a number
            ]
        }
        
        # Valid ranges for different stats
        self.valid_ranges = {
            'gold': (0, 999),
            'health': (0, 100),
            'level': (1, 9),
            'stage': (1, 7),
            'round': (1, 7),
            'experience': (0, 100)
        }
        
        logger.info("Number parser initialized")
    
    def correct_ocr_text(self, text: str) -> str:
        """Apply common OCR corrections"""
        corrected = text
        
        # Apply character corrections
        for wrong, correct in self.ocr_corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        # Remove common noise characters
        noise_chars = ['~', '`', '^', '*', '°', '¤', '§']
        for char in noise_chars:
            corrected = corrected.replace(char, '')
        
        return corrected.strip()
    
    def parse_gold(self, text: str) -> Optional[int]:
        """Parse gold amount from OCR text"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            for pattern in self.patterns['gold']:
                match = re.search(pattern, corrected_text, re.IGNORECASE)
                if match:
                    gold_value = int(match.group(1))
                    
                    # Validate range
                    min_gold, max_gold = self.valid_ranges['gold']
                    if min_gold <= gold_value <= max_gold:
                        return gold_value
            
            # Fallback: try to extract any number
            numbers = re.findall(r'\d+', corrected_text)
            if numbers:
                for num_str in numbers:
                    num = int(num_str)
                    min_gold, max_gold = self.valid_ranges['gold']
                    if min_gold <= num <= max_gold:
                        return num
            
            return None
            
        except Exception as e:
            logger.debug(f"Gold parsing failed for '{text}': {e}")
            return None
    
    def parse_health(self, text: str) -> Optional[int]:
        """Parse health amount from OCR text"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            for pattern in self.patterns['health']:
                match = re.search(pattern, corrected_text, re.IGNORECASE)
                if match:
                    health_value = int(match.group(1))
                    
                    # Validate range
                    min_health, max_health = self.valid_ranges['health']
                    if min_health <= health_value <= max_health:
                        return health_value
            
            # Fallback: try to extract any reasonable number
            numbers = re.findall(r'\d+', corrected_text)
            if numbers:
                for num_str in numbers:
                    num = int(num_str)
                    min_health, max_health = self.valid_ranges['health']
                    if min_health <= num <= max_health:
                        return num
            
            return None
            
        except Exception as e:
            logger.debug(f"Health parsing failed for '{text}': {e}")
            return None
    
    def parse_level(self, text: str) -> Optional[int]:
        """Parse player level from OCR text"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            for pattern in self.patterns['level']:
                match = re.search(pattern, corrected_text, re.IGNORECASE)
                if match:
                    level_value = int(match.group(1))
                    
                    # Validate range
                    min_level, max_level = self.valid_ranges['level']
                    if min_level <= level_value <= max_level:
                        return level_value
            
            # Fallback: try to extract any reasonable number
            numbers = re.findall(r'\d+', corrected_text)
            if numbers:
                for num_str in numbers:
                    num = int(num_str)
                    min_level, max_level = self.valid_ranges['level']
                    if min_level <= num <= max_level:
                        return num
            
            return None
            
        except Exception as e:
            logger.debug(f"Level parsing failed for '{text}': {e}")
            return None
    
    def parse_stage_round(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse stage and round from OCR text"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            for pattern in self.patterns['stage_round']:
                match = re.search(pattern, corrected_text, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 2:
                        stage = int(match.group(1))
                        round_num = int(match.group(2))
                        
                        # Validate ranges
                        min_stage, max_stage = self.valid_ranges['stage']
                        min_round, max_round = self.valid_ranges['round']
                        
                        if (min_stage <= stage <= max_stage and 
                            min_round <= round_num <= max_round):
                            return (stage, round_num)
            
            return None
            
        except Exception as e:
            logger.debug(f"Stage/round parsing failed for '{text}': {e}")
            return None
    
    def parse_experience(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse current/max experience from OCR text"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            # Look for "current/max" format
            exp_match = re.search(r'(\d+)\s*/\s*(\d+)', corrected_text)
            if exp_match:
                current_exp = int(exp_match.group(1))
                max_exp = int(exp_match.group(2))
                
                min_exp, max_exp_limit = self.valid_ranges['experience']
                if (min_exp <= current_exp <= max_exp_limit and 
                    min_exp <= max_exp <= max_exp_limit and
                    current_exp <= max_exp):
                    return (current_exp, max_exp)
            
            # Look for just current experience
            for pattern in self.patterns['experience']:
                match = re.search(pattern, corrected_text, re.IGNORECASE)
                if match:
                    exp_value = int(match.group(1))
                    min_exp, max_exp_limit = self.valid_ranges['experience']
                    if min_exp <= exp_value <= max_exp_limit:
                        return (exp_value, None)  # Current only, no max
            
            return None
            
        except Exception as e:
            logger.debug(f"Experience parsing failed for '{text}': {e}")
            return None
    
    def parse_trait_count(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse trait count like '3/6' or '(2/4)'"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            # Look for patterns like "3/6", "(3/6)", "[3/6]"
            count_patterns = [
                r'[\(\[]?(\d+)\s*/\s*(\d+)[\)\]]?',
                r'(\d+)\s*of\s*(\d+)',
                r'(\d+)\s*:\s*(\d+)'
            ]
            
            for pattern in count_patterns:
                match = re.search(pattern, corrected_text)
                if match:
                    current = int(match.group(1))
                    required = int(match.group(2))
                    
                    # Validate reasonable trait counts
                    if 0 <= current <= 10 and 1 <= required <= 10 and current <= required:
                        return (current, required)
            
            return None
            
        except Exception as e:
            logger.debug(f"Trait count parsing failed for '{text}': {e}")
            return None
    
    def parse_champion_cost(self, text: str) -> Optional[int]:
        """Parse champion cost (1-5)"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            # Look for cost indicators
            cost_patterns = [
                r'cost[:]\s*(\d+)',     # "Cost: 3"
                r'(\d+)\s*cost',        # "3 cost"
                r'tier\s*(\d+)',        # "Tier 3"
                r'^\s*(\d+)\s*$'        # Just a number
            ]
            
            for pattern in cost_patterns:
                match = re.search(pattern, corrected_text, re.IGNORECASE)
                if match:
                    cost = int(match.group(1))
                    if 1 <= cost <= 5:  # Valid TFT champion costs
                        return cost
            
            return None
            
        except Exception as e:
            logger.debug(f"Champion cost parsing failed for '{text}': {e}")
            return None
    
    def parse_item_count(self, text: str) -> Optional[int]:
        """Parse number of items on a champion"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            # Look for item count indicators
            numbers = re.findall(r'\d+', corrected_text)
            if numbers:
                for num_str in numbers:
                    num = int(num_str)
                    if 0 <= num <= 3:  # Champions can have 0-3 items
                        return num
            
            return None
            
        except Exception as e:
            logger.debug(f"Item count parsing failed for '{text}': {e}")
            return None
    
    def validate_parsed_number(self, number: int, stat_type: str) -> bool:
        """Validate if parsed number is reasonable for stat type"""
        try:
            if stat_type in self.valid_ranges:
                min_val, max_val = self.valid_ranges[stat_type]
                return min_val <= number <= max_val
            
            # Default validation for unknown types
            return 0 <= number <= 9999
            
        except Exception:
            return False
    
    def extract_all_numbers(self, text: str) -> List[int]:
        """Extract all valid numbers from text"""
        try:
            corrected_text = self.correct_ocr_text(text)
            number_strings = re.findall(r'\d+', corrected_text)
            
            numbers = []
            for num_str in number_strings:
                try:
                    num = int(num_str)
                    if 0 <= num <= 9999:  # Reasonable range
                        numbers.append(num)
                except ValueError:
                    continue
            
            return numbers
            
        except Exception as e:
            logger.debug(f"Number extraction failed for '{text}': {e}")
            return []
    
    def parse_percentage(self, text: str) -> Optional[float]:
        """Parse percentage values"""
        try:
            corrected_text = self.correct_ocr_text(text)
            
            # Look for percentage patterns
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', corrected_text)
            if percent_match:
                percentage = float(percent_match.group(1))
                if 0 <= percentage <= 100:
                    return percentage
            
            return None
            
        except Exception as e:
            logger.debug(f"Percentage parsing failed for '{text}': {e}")
            return None
    
    def smart_parse(self, text: str, context: str = None) -> Dict[str, Any]:
        """Smart parse that tries multiple parsing methods"""
        try:
            results = {}
            
            # Try context-specific parsing first
            if context:
                if context == 'gold':
                    results['gold'] = self.parse_gold(text)
                elif context == 'health':
                    results['health'] = self.parse_health(text)
                elif context == 'level':
                    results['level'] = self.parse_level(text)
                elif context == 'stage_round':
                    stage_round = self.parse_stage_round(text)
                    if stage_round:
                        results['stage'], results['round'] = stage_round
                elif context == 'experience':
                    exp_data = self.parse_experience(text)
                    if exp_data:
                        results['current_exp'], results['max_exp'] = exp_data
                elif context == 'trait':
                    trait_data = self.parse_trait_count(text)
                    if trait_data:
                        results['current_count'], results['required_count'] = trait_data
            
            # Always try general number extraction
            results['all_numbers'] = self.extract_all_numbers(text)
            results['percentage'] = self.parse_percentage(text)
            
            # Add confidence based on number of successful parses
            successful_parses = sum(1 for v in results.values() if v is not None)
            results['confidence'] = min(1.0, successful_parses / 3)
            
            return results
            
        except Exception as e:
            logger.error(f"Smart parsing failed for '{text}': {e}")
            return {'confidence': 0.0}
    
    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get statistics about parsing performance"""
        return {
            'supported_patterns': list(self.patterns.keys()),
            'valid_ranges': self.valid_ranges,
            'correction_rules': len(self.ocr_corrections)
        }