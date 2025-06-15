"""
LPOL Code Generation - Génération de Code Révolutionnaire
Utilise LPOL pour générer du code basé sur l'expérience

Copyright © 2025 Amega Mike - Proprietary License
"""

import torch
import ast
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from neural.lpol_neural_core import LPOLModel, LPOLConfig
from multimodal.text_generation import LPOLTextGenerator, GenerationConfig

@dataclass 
class CodeGenerationConfig(GenerationConfig):
    """Configuration spécialisée pour génération de code"""
    
    # Code spécifique
    language: str = "python"
    max_line_length: int = 80
    indent_size: int = 4
    
    # Validation code
    syntax_check: bool = True
    style_check: bool = True
    
    # Patterns code LPOL
    function_pattern_weight: float = 0.4
    class_pattern_weight: float = 0.3
    import_pattern_weight: float = 0.2
    
    # Guidage sémantique
    variable_consistency: bool = True
    type_hints: bool = True
    docstring_generation: bool = True

class LPOLCodeGenerator(LPOLTextGenerator):
    """Générateur de code basé sur LPOL"""
    
    def __init__(self, model: LPOLModel, tokenizer, config: CodeGenerationConfig = None):
        # Configuration spécialisée pour code
        if config is None:
            config = CodeGenerationConfig()
        
        super().__init__(model, tokenizer, config)
        
        # Patterns de code spécialisés
        self.code_patterns = {
            'function_def': ['def ', 'function ', 'fn '],
            'class_def': ['class ', 'class '],
            'import_stmt': ['import ', 'from ', 'require'],
            'control_flow': ['if ', 'for ', 'while ', 'try ', 'with '],
            'operators': ['=', '==', '!=', '<=', '>=', '+', '-', '*', '/'],
            'brackets': ['(', ')', '[', ']', '{', '}'],
            'keywords': ['return', 'yield', 'break', 'continue', 'pass']
        }
        
        # Historique patterns code réussis
        self.successful_code_patterns = []
        
        # Métriques spécifiques au code
        self.code_metrics = {
            'syntax_errors': 0,
            'successful_compilations': 0,
            'pattern_reuse_rate': 0.0,
            'avg_code_quality': 0.0
        }
    
    def generate_code(self, problem_description: str, **kwargs) -> Dict[str, Any]:
        """
        Génère du code à partir d'une description de problème
        
        Args:
            problem_description: Description du problème à résoudre
            **kwargs: Paramètres de génération
        
        Returns:
            Dictionnaire avec code généré et métadonnées LPOL
        """
        
        # Formatage prompt pour génération code
        code_prompt = self._format_code_prompt(problem_description)
        
        # Génération avec adaptation pour code
        result = self.generate(code_prompt, **kwargs)
        
        # Post-traitement spécialisé code
        processed_result = self._post_process_code(result)
        
        # Validation et scoring
        validation_result = self._validate_generated_code(processed_result['generated_text'])
        
        # Mise à jour métriques code
        self._update_code_metrics(validation_result)
        
        # Apprentissage LPOL des patterns code réussis
        if validation_result['is_valid']:
            self._learn_code_patterns(problem_description, processed_result['generated_text'])
        
        return {
            **processed_result,
            'problem_description': problem_description,
            'code_validation': validation_result,
            'code_quality_score': validation_result.get('quality_score', 0.0),
            'patterns_detected': self._detect_code_patterns(processed_result['generated_text']),
            'suggestions': validation_result.get('suggestions', [])
        }
    
    def _format_code_prompt(self, problem_description: str) -> str:
        """Formate le prompt pour optimiser la génération de code"""
        
        # Détection du type de problème
        problem_type = self._detect_problem_type(problem_description)
        
        # Templates basés sur l'expérience LPOL
        if problem_type == 'function':
            prompt = f"# Problème: {problem_description}\n\ndef solve_problem():\n    \"\"\"\n    Résout: {problem_description}\n    \"\"\"\n    "
        
        elif problem_type == 'class':
            prompt = f"# Problème: {problem_description}\n\nclass Solution:\n    \"\"\"\n    Classe pour: {problem_description}\n    \"\"\"\n    \n    def __init__(self):\n        "
        
        elif problem_type == 'algorithm':
            prompt = f"# Problème: {problem_description}\n# Algorithme optimisé\n\ndef algorithm():\n    "
        
        else:
            prompt = f"# Problème: {problem_description}\n\n"
        
        return prompt
    
    def _detect_problem_type(self, description: str) -> str:
        """Détecte le type de problème à partir de la description"""
        
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['fonction', 'function', 'calculer', 'compute']):
            return 'function'
        elif any(word in description_lower for word in ['classe', 'class', 'objet', 'object']):
            return 'class'
        elif any(word in description_lower for word in ['algorithme', 'algorithm', 'trier', 'sort', 'recherche', 'search']):
            return 'algorithm'
        else:
            return 'general'
    
    def _post_process_code(self, generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-traitement spécialisé pour le code généré"""
        
        code = generation_result['generated_text']
        
        # Nettoyage et formatage
        cleaned_code = self._clean_generated_code(code)
        
        # Ajout imports manquants si détectés
        enhanced_code = self._add_missing_imports(cleaned_code)
        
        # Formatage indentation
        formatted_code = self._format_indentation(enhanced_code)
        
        # Mise à jour du résultat
        result = generation_result.copy()
        result['generated_text'] = formatted_code
        result['original_code'] = code
        result['formatting_applied'] = True
        
        return result
    
    def _clean_generated_code(self, code: str) -> str:
        """Nettoie le code généré"""
        
        # Suppression des répétitions
        lines = code.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            stripped = line.strip()
            
            # Éviter répétitions consécutives
            if stripped != prev_line.strip() or not stripped:
                cleaned_lines.append(line)
            
            prev_line = line
        
        # Suppression lignes vides en excès
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def _add_missing_imports(self, code: str) -> str:
        """Ajoute les imports manquants détectés"""
        
        needed_imports = []
        
        # Détection imports nécessaires
        if 're.' in code or 'regex' in code.lower():
            needed_imports.append('import re')
        
        if 'np.' in code or 'numpy' in code.lower():
            needed_imports.append('import numpy as np')
        
        if 'pd.' in code or 'pandas' in code.lower():
            needed_imports.append('import pandas as pd')
        
        if 'json.' in code or 'loads' in code or 'dumps' in code:
            needed_imports.append('import json')
        
        if 'os.' in code or 'path' in code:
            needed_imports.append('import os')
        
        # Ajout en début de code
        if needed_imports:
            imports_section = '\n'.join(needed_imports) + '\n\n'
            return imports_section + code
        
        return code
    
    def _format_indentation(self, code: str) -> str:
        """Formate l'indentation du code"""
        
        lines = code.split('\n')
        formatted_lines = []
        current_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Calcul niveau indentation
            if stripped.endswith(':'):
                formatted_line = ' ' * (current_indent * self.config.indent_size) + stripped
                formatted_lines.append(formatted_line)
                current_indent += 1
            
            elif stripped in ['pass', 'break', 'continue', 'return'] or stripped.startswith('return '):
                formatted_line = ' ' * (current_indent * self.config.indent_size) + stripped
                formatted_lines.append(formatted_line)
            
            else:
                # Décrément pour certains mots-clés
                if stripped.startswith(('else:', 'elif ', 'except:', 'finally:')):
                    current_indent = max(0, current_indent - 1)
                    formatted_line = ' ' * (current_indent * self.config.indent_size) + stripped
                    current_indent += 1
                else:
                    formatted_line = ' ' * (current_indent * self.config.indent_size) + stripped
                
                formatted_lines.append(formatted_line)
        
        return '\n'.join(formatted_lines)
    
    def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """Valide le code généré"""
        
        validation_result = {
            'is_valid': False,
            'syntax_errors': [],
            'quality_score': 0.0,
            'suggestions': [],
            'metrics': {}
        }
        
        # Test syntaxe Python
        if self.config.syntax_check:
            try:
                ast.parse(code)
                validation_result['is_valid'] = True
                validation_result['syntax_errors'] = []
            except SyntaxError as e:
                validation_result['syntax_errors'].append(str(e))
        
        # Calcul score qualité
        quality_score = self._calculate_code_quality(code)
        validation_result['quality_score'] = quality_score
        
        # Suggestions d'amélioration
        suggestions = self._generate_code_suggestions(code)
        validation_result['suggestions'] = suggestions
        
        # Métriques code
        metrics = self._calculate_code_metrics(code)
        validation_result['metrics'] = metrics
        
        return validation_result
    
    def _calculate_code_quality(self, code: str) -> float:
        """Calcule un score de qualité du code (0-1)"""
        
        score = 0.0
        max_score = 0.0
        
        # Critères qualité
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 1. Longueur lignes appropriée
        max_score += 0.2
        long_lines = sum(1 for line in non_empty_lines if len(line) > self.config.max_line_length)
        if long_lines == 0:
            score += 0.2
        else:
            score += 0.2 * (1 - long_lines / len(non_empty_lines))
        
        # 2. Présence docstrings
        max_score += 0.2
        if '"""' in code or "'''" in code:
            score += 0.2
        
        # 3. Noms variables significatifs
        max_score += 0.2
        var_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        meaningful_names = sum(1 for name in var_names if len(name) > 2 and not name.islower())
        if var_names:
            score += 0.2 * (meaningful_names / len(var_names))
        
        # 4. Structure (fonctions, classes)
        max_score += 0.2
        if 'def ' in code:
            score += 0.1
        if 'class ' in code:
            score += 0.1
        
        # 5. Commentaires
        max_score += 0.2
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if comment_lines > 0:
            score += 0.2 * min(1.0, comment_lines / max(1, len(non_empty_lines) / 5))
        
        return score / max_score if max_score > 0 else 0.0
    
    def _generate_code_suggestions(self, code: str) -> List[str]:
        """Génère des suggestions d'amélioration"""
        
        suggestions = []
        lines = code.split('\n')
        
        # Lignes trop longues
        long_lines = [i for i, line in enumerate(lines) if len(line) > self.config.max_line_length]
        if long_lines:
            suggestions.append(f"Raccourcir les lignes {long_lines} (> {self.config.max_line_length} caractères)")
        
        # Manque docstrings
        if 'def ' in code and '"""' not in code:
            suggestions.append("Ajouter des docstrings aux fonctions")
        
        # Variables à une lettre
        single_char_vars = re.findall(r'\b[a-z]\b', code)
        if single_char_vars:
            suggestions.append("Utiliser des noms de variables plus explicites")
        
        # Imports manquants détectés
        if 'np.' in code and 'import numpy' not in code:
            suggestions.append("Ajouter 'import numpy as np'")
        
        return suggestions
    
    def _calculate_code_metrics(self, code: str) -> Dict[str, Any]:
        """Calcule des métriques détaillées du code"""
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': sum(1 for line in lines if line.strip().startswith('#')),
            'functions': code.count('def '),
            'classes': code.count('class '),
            'complexity_estimate': len(re.findall(r'\b(if|for|while|try|except)\b', code)),
            'avg_line_length': sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
        }
    
    def _detect_code_patterns(self, code: str) -> Dict[str, int]:
        """Détecte les patterns de code utilisés"""
        
        patterns_found = {}
        
        for pattern_type, keywords in self.code_patterns.items():
            count = sum(code.count(keyword) for keyword in keywords)
            if count > 0:
                patterns_found[pattern_type] = count
        
        return patterns_found
    
    def _learn_code_patterns(self, problem: str, successful_code: str):
        """Apprend des patterns de code réussis pour LPOL"""
        
        # Extraction pattern réussi
        pattern = {
            'problem_type': self._detect_problem_type(problem),
            'code_structure': self._analyze_code_structure(successful_code),
            'patterns_used': self._detect_code_patterns(successful_code),
            'quality_score': self._calculate_code_quality(successful_code)
        }
        
        # Stockage pour réutilisation
        self.successful_code_patterns.append(pattern)
        
        # Maintenir taille raisonnable
        if len(self.successful_code_patterns) > 100:
            # Garder les meilleurs patterns
            self.successful_code_patterns.sort(key=lambda p: p['quality_score'], reverse=True)
            self.successful_code_patterns = self.successful_code_patterns[:100]
    
    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyse la structure du code pour pattern learning"""
        
        return {
            'has_functions': 'def ' in code,
            'has_classes': 'class ' in code,
            'has_imports': any(line.strip().startswith(('import ', 'from ')) for line in code.split('\n')),
            'control_structures': len(re.findall(r'\b(if|for|while|try)\b', code)),
            'line_count': len([line for line in code.split('\n') if line.strip()]),
            'indentation_levels': max(len(line) - len(line.lstrip()) for line in code.split('\n')) // 4
        }
    
    def _update_code_metrics(self, validation_result: Dict[str, Any]):
        """Met à jour les métriques de génération de code"""
        
        if validation_result['is_valid']:
            self.code_metrics['successful_compilations'] += 1
        else:
            self.code_metrics['syntax_errors'] += 1
        
        # Moyenne mobile qualité
        quality = validation_result.get('quality_score', 0.0)
        alpha = 0.1
        self.code_metrics['avg_code_quality'] = (
            alpha * quality + (1 - alpha) * self.code_metrics['avg_code_quality']
        )
        
        # Taux réutilisation patterns
        total_generations = self.code_metrics['successful_compilations'] + self.code_metrics['syntax_errors']
        if total_generations > 0:
            self.code_metrics['pattern_reuse_rate'] = len(self.successful_code_patterns) / total_generations
    
    def get_code_generation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de génération de code"""
        
        base_stats = self.get_generation_stats()
        
        return {
            **base_stats,
            **self.code_metrics,
            'patterns_learned': len(self.successful_code_patterns),
            'success_rate': (
                self.code_metrics['successful_compilations'] / 
                max(1, self.code_metrics['successful_compilations'] + self.code_metrics['syntax_errors'])
            )
        }

def demo_lpol_code_generation():
    """Démonstration génération de code LPOL"""
    
    print("💻 LPOL Code Generation Demo")
    print("=" * 40)
    
    # Configuration pour code
    from neural.lpol_neural_core import get_default_config
    from multimodal.text_generation import LPOLTokenizer
    
    model_config = get_default_config()
    model_config.vocab_size = 8000  # Plus grand pour code
    
    # Modèle et générateur
    model = LPOLModel(model_config)
    tokenizer = LPOLTokenizer(model_config.vocab_size)
    
    code_config = CodeGenerationConfig(
        max_length=300,
        temperature=0.7,
        syntax_check=True,
        style_check=True
    )
    
    generator = LPOLCodeGenerator(model, tokenizer, code_config)
    
    # Problèmes de code à résoudre
    code_problems = [
        "Écrire une fonction qui trouve le maximum d'une liste",
        "Créer une classe pour représenter un point 2D",
        "Implémenter un algorithme de tri rapide",
        "Fonction pour calculer la factorielle d'un nombre"
    ]
    
    print("🚀 Génération de code en cours...\n")
    
    for i, problem in enumerate(code_problems, 1):
        print(f"📝 Problème {i}: {problem}")
        print("-" * 50)
        
        result = generator.generate_code(problem)
        
        print(f"Code généré:")
        print("```python")
        print(result['generated_text'])
        print("```")
        
        print(f"\n📊 Validation:")
        validation = result['code_validation']
        print(f"  Syntaxe valide: {'✅' if validation['is_valid'] else '❌'}")
        print(f"  Score qualité: {validation['quality_score']:.2f}")
        
        if validation['suggestions']:
            print(f"  Suggestions: {', '.join(validation['suggestions'][:2])}")
        
        print(f"\n🧠 LPOL Metadata:")
        print(f"  Confiance: {result['confidence']:.3f}")
        print(f"  Patterns: {result['patterns_used']:.1f}")
        print(f"  Temps: {result['generation_time']:.3f}s")
        
        print(f"\n💻 Patterns détectés: {result['patterns_detected']}")
        print("=" * 60)
    
    # Statistiques finales
    stats = generator.get_code_generation_stats()
    print(f"\n📈 Statistiques génération code:")
    print(f"  Générations: {stats['total_generations']}")
    print(f"  Succès syntaxe: {stats['successful_compilations']}")
    print(f"  Taux réussite: {stats['success_rate']:.2f}")
    print(f"  Qualité moyenne: {stats['avg_code_quality']:.2f}")
    print(f"  Patterns appris: {stats['patterns_learned']}")

if __name__ == "__main__":
    demo_lpol_code_generation()