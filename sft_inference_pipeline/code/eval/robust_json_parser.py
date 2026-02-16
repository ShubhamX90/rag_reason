#!/usr/bin/env python3
"""Robust JSON Parser with Recovery"""
import json
import re
from typing import Dict, Any

def parse_json_robust(text: str, expected_type='auto'):
    """Robust JSON parser with automatic recovery."""
    text = text.strip()
    if not text:
        return {'success': False, 'data': None, 'error': 'Empty', 'partial': None, 'repairs': []}
    
    repairs = []
    
    # Strategy 1: Try as-is
    try:
        return {'success': True, 'data': json.loads(text), 'error': None, 'partial': None, 'repairs': []}
    except json.JSONDecodeError as e:
        orig_error = str(e)
    
    # Strategy 2: Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    repairs.append('removed_trailing_commas')
    try:
        return {'success': True, 'data': json.loads(text), 'error': None, 'partial': None, 'repairs': repairs}
    except:
        pass
    
    # Strategy 3: Fix incomplete structures
    open_sq = text.count('[')
    close_sq = text.count(']')
    open_cu = text.count('{')
    close_cu = text.count('}')
    
    if open_sq > close_sq:
        text += ']' * (open_sq - close_sq)
        repairs.append('added_closing_brackets')
    if open_cu > close_cu:
        text += '}' * (open_cu - close_cu)
        repairs.append('added_closing_braces')
    
    try:
        return {'success': True, 'data': json.loads(text), 'error': None, 'partial': None, 'repairs': repairs}
    except:
        pass
    
    # Strategy 4: Fix empty source_quality (common issue)
    text = re.sub(r'"source_quality":\s*""', '"source_quality": "low"', text)
    repairs.append('fixed_empty_source_quality')
    try:
        return {'success': True, 'data': json.loads(text), 'error': None, 'partial': None, 'repairs': repairs}
    except:
        pass
    
    # Strategy 5: Extract valid array elements (partial recovery)
    if text.lstrip().startswith('['):
        elements = extract_valid_array_elements(text)
        if elements:
            repairs.append('partial_array_recovery')
            return {'success': False, 'data': None, 'error': orig_error, 'partial': elements, 'repairs': repairs}
    
    return {'success': False, 'data': None, 'error': orig_error, 'partial': None, 'repairs': repairs}


def extract_valid_array_elements(text: str):
    """Extract complete objects from potentially broken array."""
    elements = []
    depth = 0
    current = ""
    in_string = False
    escape = False
    
    # Skip opening bracket
    start = text.find('[')
    if start == -1:
        return None
    
    for char in text[start+1:]:
        if escape:
            current += char
            escape = False
            continue
        
        if char == '\\' and in_string:
            current += char
            escape = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
            current += char
        elif char == '"' and in_string:
            in_string = False
            current += char
        elif not in_string:
            if char == '{':
                depth += 1
                current += char
            elif char == '}':
                depth -= 1
                current += char
                if depth == 0 and current.strip():
                    try:
                        elem = json.loads(current.strip())
                        elements.append(elem)
                        current = ""
                    except:
                        current = ""
            elif char == ',' and depth == 0:
                continue
            elif char == ']' and depth == 0:
                break
            else:
                current += char
        else:
            current += char
    
    return elements if elements else None
