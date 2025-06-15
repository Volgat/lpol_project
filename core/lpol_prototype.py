import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Problem:
    """Représente un problème à résoudre"""
    id: str
    description: str
    input_data: Any
    expected_output: Any
    difficulty: int  # 1-10
    category: str
    
@dataclass
class Solution:
    """Représente une solution proposée"""
    code: str
    reasoning: str
    confidence: float
    execution_time: float = 0.0

@dataclass
class Feedback:
    """Feedback sur une solution"""
    is_correct: bool
    score: float  # 0-1
    error_details: str
    improvement_suggestions: List[str]
    learned_patterns: List[str]

class LPOLLearner:
    """Module principal d'apprentissage LPOL"""
    
    def __init__(self):
        self.memory = LPOLMemoryAbstraction()
        self.checker = LPOLChecker()
        self.reasoner = LPOLReasoner()
        self.solved_problems = []
        self.failed_attempts = []
        self.learning_patterns = {}
        
    def solve_problem(self, problem: Problem) -> Solution:
        """Tente de résoudre un problème en utilisant l'expérience passée"""
        
        # 1. Chercher des patterns similaires dans la mémoire
        similar_patterns = self.memory.find_similar_patterns(problem)
        
        # 2. Générer une solution basée sur l'expérience
        solution = self._generate_solution(problem, similar_patterns)
        
        # 3. Auto-vérification avec le reasoner
        reasoning_check = self.reasoner.verify_logic(solution, problem)
        
        return solution
    
    def learn_from_feedback(self, problem: Problem, solution: Solution, feedback: Feedback):
        """Apprentissage à partir du feedback reçu"""
        
        if feedback.is_correct:
            # Succès : abstraire et mémoriser le pattern gagnant
            pattern = self._extract_successful_pattern(problem, solution)
            self.memory.store_pattern(pattern)
            self.solved_problems.append((problem, solution, feedback))
        else:
            # Échec : analyser l'erreur et apprendre
            error_analysis = self._analyze_failure(problem, solution, feedback)
            self.memory.store_failure_lesson(error_analysis)
            self.failed_attempts.append((problem, solution, feedback))
            
        # Mise à jour des patterns d'apprentissage
        self._update_learning_patterns(problem, solution, feedback)
    
    def _generate_solution(self, problem: Problem, similar_patterns: List[Dict]) -> Solution:
        """Génère une solution basée sur les patterns connus"""
        
        if similar_patterns:
            # Adaptation d'un pattern existant
            base_pattern = similar_patterns[0]
            adapted_code = self._adapt_pattern_to_problem(base_pattern, problem)
            confidence = 0.8
        else:
            # Nouvelle exploration
            adapted_code = self._explore_new_solution(problem)
            confidence = 0.3
            
        reasoning = f"Approche basée sur {len(similar_patterns)} patterns similaires"
        
        return Solution(
            code=adapted_code,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def _extract_successful_pattern(self, problem: Problem, solution: Solution) -> Dict:
        """Extrait un pattern réutilisable d'une solution réussie"""
        return {
            'category': problem.category,
            'difficulty': problem.difficulty,
            'code_structure': self._analyze_code_structure(solution.code),
            'key_concepts': self._extract_key_concepts(problem, solution),
            'reusable_logic': solution.code
        }
    
    def _analyze_failure(self, problem: Problem, solution: Solution, feedback: Feedback) -> Dict:
        """Analyse un échec pour en tirer des leçons"""
        return {
            'problem_type': problem.category,
            'failed_approach': solution.code,
            'error_type': feedback.error_details,
            'lessons': feedback.improvement_suggestions,
            'avoid_patterns': self._identify_problematic_patterns(solution.code)
        }
    
    def _adapt_pattern_to_problem(self, pattern: Dict, problem: Problem) -> str:
        """Adapte un pattern existant au nouveau problème"""
        base_code = pattern.get('reusable_logic', '')
        return f"# Adapté du pattern {pattern.get('category', 'unknown')}\n{base_code}"
    
    def _explore_new_solution(self, problem: Problem) -> str:
        """Explore une nouvelle approche pour un problème inconnu"""
        return f"# Nouvelle exploration pour: {problem.description}\n# TODO: Implémenter la logique"
    
    def _analyze_code_structure(self, code: str) -> Dict:
        """Analyse la structure du code pour identifier des patterns"""
        return {
            'has_loops': 'for' in code or 'while' in code,
            'has_conditions': 'if' in code,
            'has_functions': 'def' in code,
            'complexity': len(code.split('\n'))
        }
    
    def _extract_key_concepts(self, problem: Problem, solution: Solution) -> List[str]:
        """Extrait les concepts clés utilisés dans la solution"""
        concepts = []
        if 'sort' in solution.code: concepts.append('sorting')
        if 'loop' in solution.code: concepts.append('iteration')
        if 'recursive' in solution.code: concepts.append('recursion')
        return concepts
    
    def _identify_problematic_patterns(self, code: str) -> List[str]:
        """Identifie les patterns problématiques à éviter"""
        problems = []
        if code.count('for') > 3: problems.append('too_many_nested_loops')
        if 'while True' in code and 'break' not in code: problems.append('infinite_loop_risk')
        return problems
    
    def _update_learning_patterns(self, problem: Problem, solution: Solution, feedback: Feedback):
        """Met à jour les patterns d'apprentissage globaux"""
        category = problem.category
        if category not in self.learning_patterns:
            self.learning_patterns[category] = {'success_rate': 0, 'attempts': 0}
        
        self.learning_patterns[category]['attempts'] += 1
        if feedback.is_correct:
            self.learning_patterns[category]['success_rate'] += 1

class LPOLChecker:
    """Module de vérification des solutions"""
    
    def evaluate_solution(self, problem: Problem, solution: Solution) -> Feedback:
        """Évalue une solution et génère un feedback"""
        
        # Simulation d'exécution et vérification
        is_correct, score, error_details = self._execute_and_check(problem, solution)
        
        # Génération de suggestions d'amélioration
        suggestions = self._generate_improvement_suggestions(problem, solution, is_correct)
        
        # Identification des patterns appris
        learned_patterns = self._identify_learned_patterns(problem, solution, is_correct)
        
        return Feedback(
            is_correct=is_correct,
            score=score,
            error_details=error_details,
            improvement_suggestions=suggestions,
            learned_patterns=learned_patterns
        )
    
    def _execute_and_check(self, problem: Problem, solution: Solution) -> tuple:
        """Simule l'exécution du code et vérifie le résultat"""
        # Simulation simplifiée pour le prototype
        
        # Vérification de base du code
        if not solution.code or solution.code.strip() == "":
            return False, 0.0, "Code vide"
        
        if "TODO" in solution.code:
            return False, 0.2, "Solution incomplète"
        
        # Simulation d'un succès partiel basé sur la confiance
        success_probability = solution.confidence * 0.7 + random.random() * 0.3
        
        if success_probability > 0.6:
            return True, success_probability, ""
        else:
            return False, success_probability, f"Erreur d'exécution simulée (prob: {success_probability:.2f})"
    
    def _generate_improvement_suggestions(self, problem: Problem, solution: Solution, is_correct: bool) -> List[str]:
        """Génère des suggestions d'amélioration"""
        suggestions = []
        
        if not is_correct:
            suggestions.append("Revoir la logique principale")
            suggestions.append("Vérifier les cas limites")
        
        if solution.confidence < 0.5:
            suggestions.append("Chercher des patterns similaires dans l'historique")
        
        if len(solution.code.split('\n')) > 20:
            suggestions.append("Simplifier le code")
        
        return suggestions
    
    def _identify_learned_patterns(self, problem: Problem, solution: Solution, is_correct: bool) -> List[str]:
        """Identifie les patterns qui peuvent être appris de cette solution"""
        patterns = []
        
        if is_correct:
            patterns.append(f"successful_approach_for_{problem.category}")
            if 'for' in solution.code:
                patterns.append("effective_iteration_pattern")
        
        return patterns

class LPOLReasoner:
    """Module de raisonnement et auto-vérification"""
    
    def verify_logic(self, solution: Solution, problem: Problem) -> Dict:
        """Vérifie la logique de la solution"""
        
        logic_check = {
            'coherence_score': self._check_coherence(solution, problem),
            'complexity_appropriate': self._check_complexity(solution, problem),
            'reasoning_quality': self._evaluate_reasoning(solution.reasoning),
            'potential_issues': self._identify_potential_issues(solution)
        }
        
        return logic_check
    
    def _check_coherence(self, solution: Solution, problem: Problem) -> float:
        """Vérifie la cohérence de la solution avec le problème"""
        # Simulation simplifiée
        if problem.description.lower() in solution.code.lower():
            return 0.8
        return 0.4
    
    def _check_complexity(self, solution: Solution, problem: Problem) -> bool:
        """Vérifie si la complexité est appropriée"""
        code_complexity = len(solution.code.split('\n'))
        return code_complexity <= problem.difficulty * 5
    
    def _evaluate_reasoning(self, reasoning: str) -> float:
        """Évalue la qualité du raisonnement"""
        if len(reasoning) > 10 and any(word in reasoning.lower() for word in ['parce', 'car', 'donc', 'ainsi']):
            return 0.7
        return 0.3
    
    def _identify_potential_issues(self, solution: Solution) -> List[str]:
        """Identifie les problèmes potentiels"""
        issues = []
        
        if solution.confidence < 0.3:
            issues.append("Confiance très faible")
        
        if 'TODO' in solution.code:
            issues.append("Code incomplet")
        
        return issues

class LPOLMemoryAbstraction:
    """Base de connaissances dynamique et adaptative"""
    
    def __init__(self):
        self.successful_patterns = []
        self.failure_lessons = []
        self.pattern_usage_stats = {}
    
    def store_pattern(self, pattern: Dict):
        """Stocke un pattern réussi"""
        self.successful_patterns.append(pattern)
        pattern_id = pattern.get('category', 'unknown')
        self.pattern_usage_stats[pattern_id] = self.pattern_usage_stats.get(pattern_id, 0) + 1
    
    def store_failure_lesson(self, lesson: Dict):
        """Stocke une leçon d'échec"""
        self.failure_lessons.append(lesson)
    
    def find_similar_patterns(self, problem: Problem) -> List[Dict]:
        """Trouve des patterns similaires au problème actuel"""
        similar_patterns = []
        
        for pattern in self.successful_patterns:
            similarity_score = self._calculate_similarity(problem, pattern)
            if similarity_score > 0.5:
                similar_patterns.append({
                    **pattern,
                    'similarity_score': similarity_score
                })
        
        # Tri par similarité décroissante
        similar_patterns.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_patterns[:3]  # Top 3 patterns similaires
    
    def _calculate_similarity(self, problem: Problem, pattern: Dict) -> float:
        """Calcule la similarité entre un problème et un pattern"""
        similarity = 0.0
        
        # Similarité de catégorie
        if problem.category == pattern.get('category'):
            similarity += 0.5
        
        # Similarité de difficulté
        difficulty_diff = abs(problem.difficulty - pattern.get('difficulty', 5))
        similarity += (10 - difficulty_diff) / 10 * 0.3
        
        # Utilisation passée du pattern
        usage_count = self.pattern_usage_stats.get(pattern.get('category', ''), 0)
        similarity += min(usage_count / 10, 0.2)
        
        return similarity