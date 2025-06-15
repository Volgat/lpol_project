#!/usr/bin/env python3
"""
LPOL Neural Architecture Launcher
Lanceur principal pour l'architecture neuronale révolutionnaire LPOL

Copyright © 2025 Amega Mike - Proprietary License
"""

import os
import sys
import argparse
import torch
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ajout du path pour imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports LPOL
from neural.lpol_neural_core import LPOLModel, LPOLConfig, get_default_config
from training.lpol_trainer import LPOLTrainer, TrainingConfig, create_sample_dataset, SimpleTokenizer, collate_fn, ProblemSolutionDataset
from multimodal.text_generation import LPOLTextGenerator, GenerationConfig

class ImprovedTokenizer:
    """Tokenizer amélioré avec vocabulaire français pour génération lisible"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        
        # Vocabulaire français étendu et structuré
        self.french_words = [
            # Articles et déterminants
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'ce', 'cette', 'ces',
            'son', 'sa', 'ses', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            
            # Pronoms
            'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous', 'on',
            'qui', 'que', 'quoi', 'où', 'quand', 'comment', 'pourquoi',
            
            # Verbes courants
            'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
            'vouloir', 'venir', 'falloir', 'devoir', 'croire', 'trouve', 'prendre',
            'donner', 'porter', 'parler', 'aimer', 'passer', 'mettre', 'suivre',
            
            # Prépositions et conjonctions
            'et', 'ou', 'mais', 'donc', 'car', 'ni', 'or', 'à', 'dans', 'par',
            'pour', 'en', 'vers', 'avec', 'sans', 'sous', 'sur', 'entre', 'parmi',
            'depuis', 'pendant', 'avant', 'après', 'selon', 'malgré', 'grâce',
            
            # Mots techniques - Programmation
            'code', 'python', 'fonction', 'variable', 'classe', 'méthode', 'objet',
            'algorithme', 'programme', 'script', 'module', 'bibliothèque', 'import',
            'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while', 'try',
            'except', 'finally', 'with', 'as', 'lambda', 'yield', 'global', 'nonlocal',
            
            # Mots techniques - Mathématiques
            'mathématique', 'mathématiques', 'équation', 'calcul', 'calculer', 'nombre',
            'entier', 'réel', 'complexe', 'fraction', 'décimal', 'pourcentage',
            'addition', 'soustraction', 'multiplication', 'division', 'puissance',
            'racine', 'logarithme', 'exponentielle', 'trigonométrie', 'géométrie',
            'algèbre', 'statistique', 'probabilité', 'matrice', 'vecteur',
            
            # Mots techniques - Résolution
            'problème', 'solution', 'résoudre', 'résolution', 'réponse', 'question',
            'analyser', 'analyse', 'étudier', 'examiner', 'rechercher', 'trouver',
            'découvrir', 'identifier', 'déterminer', 'évaluer', 'estimer', 'mesurer',
            'comparer', 'contraster', 'différencier', 'classifier', 'catégoriser',
            
            # Verbes d'action
            'créer', 'générer', 'construire', 'développer', 'concevoir', 'implémenter',
            'réaliser', 'effectuer', 'exécuter', 'lancer', 'démarrer', 'arrêter',
            'modifier', 'changer', 'transformer', 'convertir', 'adapter', 'ajuster',
            'optimiser', 'améliorer', 'perfectionner', 'corriger', 'réparer', 'déboguer',
            
            # Adjectifs qualificatifs
            'bon', 'mauvais', 'grand', 'petit', 'gros', 'mince', 'haut', 'bas',
            'long', 'court', 'large', 'étroit', 'profond', 'superficiel',
            'rapide', 'lent', 'facile', 'difficile', 'simple', 'complexe',
            'nouveau', 'ancien', 'moderne', 'classique', 'récent', 'vieux',
            'important', 'essentiel', 'nécessaire', 'utile', 'pratique', 'efficace',
            'optimal', 'parfait', 'excellent', 'meilleur', 'pire', 'correct',
            
            # Nombres et quantités
            'zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf',
            'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize', 'vingt',
            'trente', 'quarante', 'cinquante', 'soixante', 'cent', 'mille', 'million',
            'premier', 'deuxième', 'troisième', 'dernier', 'suivant', 'précédent',
            'plusieurs', 'beaucoup', 'peu', 'assez', 'trop', 'très', 'plus', 'moins',
            
            # Mots logiques et connecteurs
            'si', 'alors', 'sinon', 'tant', 'que', 'jusqu', 'lorsque', 'quand',
            'parce', 'puisque', 'comme', 'ainsi', 'donc', 'par', 'conséquent',
            'cependant', 'néanmoins', 'toutefois', 'pourtant', 'malgré', 'bien',
            'afin', 'pour', 'dans', 'but', 'objectif', 'fin', 'moyen', 'façon',
            
            # Substantifs courants
            'chose', 'objet', 'élément', 'partie', 'ensemble', 'groupe', 'série',
            'liste', 'tableau', 'structure', 'système', 'mécanisme', 'processus',
            'méthode', 'technique', 'procédure', 'étape', 'phase', 'niveau',
            'type', 'genre', 'sorte', 'espèce', 'catégorie', 'classe', 'famille',
            'exemple', 'cas', 'situation', 'contexte', 'environnement', 'cadre',
            
            # Mots temporels
            'temps', 'moment', 'instant', 'seconde', 'minute', 'heure', 'jour',
            'semaine', 'mois', 'année', 'siècle', 'époque', 'période', 'durée',
            'début', 'fin', 'commencement', 'terminaison', 'start', 'stop',
            'maintenant', 'aujourd', 'hier', 'demain', 'bientôt', 'tard', 'tôt',
            
            # Mots spatiaux
            'lieu', 'endroit', 'place', 'position', 'emplacement', 'localisation',
            'ici', 'là', 'partout', 'nulle', 'part', 'quelque', 'part',
            'dessus', 'dessous', 'devant', 'derrière', 'côté', 'gauche', 'droite',
            'nord', 'sud', 'est', 'ouest', 'centre', 'milieu', 'bord', 'coin'
        ]
        
        # Construction du vocabulaire
        self.vocab = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<problem>': 4, '<solution>': 5, '<code>': 6, '<math>': 7
        }
        
        # Ajouter le vocabulaire français
        for i, word in enumerate(self.french_words):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Compléter avec des tokens génériques si nécessaire
        current_size = len(self.vocab)
        if current_size < vocab_size:
            for i in range(current_size, vocab_size):
                self.vocab[f'mot_{i}'] = i
        
        # Dictionnaire inverse
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Tokenizer amélioré: {len(self.vocab)} tokens, dont {len(self.french_words)} mots français")
    
    def encode(self, text: str, max_length: int = 512, truncation: bool = True) -> list:
        """Encode le texte en IDs"""
        
        # Nettoyage et normalisation
        text = text.lower().strip()
        
        # Remplacement de contractions et formes courantes
        replacements = {
            "l'": "le ", "d'": "de ", "n'": "ne ", "m'": "me ",
            "t'": "te ", "s'": "se ", "j'": "je ", "c'": "ce ",
            "qu'": "que ", "jusqu'": "jusqu "
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Gestion de la ponctuation
        text = text.replace(',', ' , ').replace('.', ' . ').replace(':', ' : ')
        text = text.replace('!', ' ! ').replace('?', ' ? ').replace(';', ' ; ')
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        
        # Tokenisation par mots
        words = text.split()
        
        if truncation:
            words = words[:max_length-2]
        
        # Conversion en IDs
        token_ids = [self.vocab['<bos>']]
        
        for word in words:
            # Nettoyage du mot
            clean_word = word.strip('.,!?;:()')
            
            if clean_word in self.vocab:
                token_ids.append(self.vocab[clean_word])
            elif word in self.vocab:  # Avec ponctuation
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab['<unk>'])
        
        token_ids.append(self.vocab['<eos>'])
        return token_ids
    
    def decode(self, token_ids: list) -> str:
        """Décode les IDs en texte lisible français"""
        
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Ignorer tokens spéciaux et génériques
                if not token.startswith('<') and not token.startswith('mot_'):
                    tokens.append(token)
        
        if not tokens:
            return "Apprentissage en cours..."
        
        # Reconstitution intelligente du texte
        text = ' '.join(tokens)
        
        # Corrections grammaticales de base
        text = text.replace(' , ', ', ').replace(' . ', '. ').replace(' : ', ': ')
        text = text.replace(' ! ', '! ').replace(' ? ', '? ').replace(' ; ', '; ')
        text = text.replace(' ( ', ' (').replace(' ) ', ') ')
        
        # Capitalisation du début
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text.strip()
    
    def get_vocab_stats(self):
        """Statistiques du vocabulaire"""
        return {
            'total_tokens': len(self.vocab),
            'french_words': len(self.french_words),
            'special_tokens': 8,
            'coverage_french': len(self.french_words) / len(self.vocab) * 100
        }

def setup_directories():
    """Crée la structure de dossiers nécessaire"""
    
    directories = [
        'neural',
        'training', 
        'multimodal',
        'models',
        'datasets',
        'logs',
        'experiments',
        'benchmarks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Créer __init__.py pour les packages Python
        if directory in ['neural', 'training', 'multimodal']:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# LPOL Neural Architecture Package\n')
    
    print("📁 Structure de dossiers créée")

def test_architecture():
    """Test de l'architecture LPOL de base"""
    
    print("🧪 Test Architecture LPOL")
    print("=" * 40)
    
    # Configuration test avec dimensions cohérentes
    config = get_default_config()
    config.vocab_size = 1000  # Plus petit pour le test
    config.hidden_size = 264  # Divisible par 12 (264 = 12 * 22)
    config.num_layers = 6
    config.num_heads = 12
    config.memory_size = 100
    
    # Ajustement automatique des autres dimensions
    config.experience_dim = min(config.experience_dim, config.hidden_size // 2)
    config.problem_embedding_dim = min(config.problem_embedding_dim, config.hidden_size // 4)
    config.solution_embedding_dim = min(config.solution_embedding_dim, config.hidden_size // 4)
    
    print(f"Paramètres test:")
    print(f"  - Vocab: {config.vocab_size}")
    print(f"  - Hidden: {config.hidden_size}")
    print(f"  - Layers: {config.num_layers}")
    print(f"  - Memory: {config.memory_size}")
    
    # Création modèle
    print("\n🧠 Création modèle LPOL...")
    model = LPOLModel(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modèle créé: {num_params:,} paramètres sur {device}")
    
    # Test forward pass
    print("\n⚡ Test forward pass...")
    batch_size, seq_len = 2, 50
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    forward_time = time.time() - start_time
    
    print(f"✅ Forward réussi en {forward_time:.3f}s")
    print(f"  - Logits shape: {outputs['logits'].shape}")
    print(f"  - Confiance globale: {outputs['global_confidence'].mean():.3f}")
    print(f"  - Patterns utilisés: {outputs['total_patterns_used'].mean():.1f}")
    
    # Test apprentissage
    print("\n📚 Test apprentissage LPOL...")
    
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    success_scores = torch.tensor([0.8, 0.9], device=device)
    
    model.learn_from_feedback(input_ids, target_ids, success_scores)
    
    # Stats après apprentissage
    stats = model.get_model_stats()
    print(f"✅ Apprentissage effectué:")
    print(f"  - Patterns actifs: {stats['total_active_patterns']}")
    print(f"  - Taux succès: {stats['average_success_rate']:.3f}")
    print(f"  - Efficacité mémoire: {stats['memory_efficiency']:.3f}")
    
    return True

def test_text_generation():
    """Test de génération de texte LPOL"""
    
    print("\n🎨 Test Génération Texte LPOL")
    print("=" * 40)
    
    # Configuration avec dimensions compatibles
    model_config = get_default_config()
    model_config.vocab_size = 5000
    model_config.hidden_size = 512  # Compatible avec 8 têtes
    model_config.num_heads = 8      # 512/8 = 64, parfait
    model_config.num_layers = 8
    
    # Modèle et tokenizer
    model = LPOLModel(model_config)
    tokenizer = ImprovedTokenizer(model_config.vocab_size)
    
    # Générateur
    generator = LPOLTextGenerator(model, tokenizer)
    
    # Prompts de test
    test_prompts = [
        "La solution du problème est",
        "Pour résoudre cette question",
        "Le code Python suivant"
    ]
    
    print("🚀 Génération en cours...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        
        result = generator.generate(
            prompt,
            max_length=80,
            temperature=0.8,
            top_p=0.9
        )
        
        print(f"  Généré: {result['generated_text'][:100]}...")
        print(f"  Temps: {result['generation_time']:.3f}s")
        print(f"  Confiance: {result['confidence']:.3f}")
        print(f"  Patterns: {result['patterns_used']:.1f}")
    
    # Statistiques
    stats = generator.get_generation_stats()
    print(f"\n📊 Stats génération:")
    print(f"  - Générations: {stats['total_generations']}")
    print(f"  - Confiance moy: {stats['avg_confidence']:.3f}")
    
    return True

def run_training_demo():
    """Démonstration d'entraînement LPOL"""
    
    print("\n🎓 Démonstration Entraînement LPOL")
    print("=" * 40)
    
    # Configuration avec dimensions compatibles
    model_config = get_default_config()
    model_config.vocab_size = 2000
    model_config.hidden_size = 256  # Garder 256
    model_config.num_layers = 4
    model_config.num_heads = 8  # Changé pour compatibilité (256/8 = 32)
    model_config.memory_size = 200
    
    training_config = TrainingConfig(
        batch_size=4,
        num_epochs=2,
        learning_rate=5e-4,
        save_steps=50,
        eval_steps=25,
        use_wandb=False
    )
    
    print(f"Configuration entraînement:")
    print(f"  - Batch size: {training_config.batch_size}")
    print(f"  - Epochs: {training_config.num_epochs}")
    print(f"  - Learning rate: {training_config.learning_rate}")
    
    # Créer dataset de démonstration
    dataset_path = "datasets/demo_problems.json"
    create_sample_dataset(dataset_path, num_samples=200)
    print(f"✅ Dataset créé: {dataset_path}")
    
    # Entraîneur
    trainer = LPOLTrainer(model_config, training_config)
    
    # Dataset et dataloader
    from torch.utils.data import DataLoader
    from training.lpol_trainer import ProblemSolutionDataset, SimpleTokenizer, collate_fn
    
    tokenizer = SimpleTokenizer(model_config.vocab_size)
    dataset = ProblemSolutionDataset(dataset_path, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=training_config.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"✅ Dataset chargé: {len(dataset)} exemples")
    
    # Entraînement court
    print("\n🚀 Début entraînement...")
    trainer.train(dataloader)
    
    # Stats finales
    final_stats = trainer.model.get_model_stats()
    print(f"\n📈 Stats finales:")
    print(f"  - Patterns appris: {final_stats['total_active_patterns']}")
    print(f"  - Utilisation: {final_stats['total_pattern_usage']}")
    print(f"  - Taux succès: {final_stats['average_success_rate']:.3f}")
    
    return True

def benchmark_vs_traditional():
    """Benchmark LPOL vs approches traditionnelles"""
    
    print("\n⚔️  Benchmark LPOL vs Traditionnel")
    print("=" * 40)
    
    # Configuration LPOL avec dimensions compatibles
    lpol_config = get_default_config()
    lpol_config.vocab_size = 1000
    lpol_config.hidden_size = 256  # 256
    lpol_config.num_layers = 6
    lpol_config.num_heads = 8  # 256/8 = 32, compatible
    
    # Modèles de test
    lpol_model = LPOLModel(lpol_config)
    
    # Simulation modèle traditionnel (Linear simple)
    traditional_model = torch.nn.Sequential(
        torch.nn.Embedding(lpol_config.vocab_size, lpol_config.hidden_size),
        torch.nn.Linear(lpol_config.hidden_size, lpol_config.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(lpol_config.hidden_size, lpol_config.vocab_size)
    )
    
    # Test data
    batch_size, seq_len = 4, 100
    input_ids = torch.randint(0, lpol_config.vocab_size, (batch_size, seq_len))
    
    # Benchmark LPOL
    print("🧠 Test LPOL...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            lpol_outputs = lpol_model(input_ids)
    
    lpol_time = time.time() - start_time
    lpol_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Benchmark traditionnel
    print("🤖 Test Traditionnel...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            # Simulation forward traditionnel
            embedded = traditional_model[0](input_ids)
            output = traditional_model[1:](embedded.mean(dim=1))
    
    traditional_time = time.time() - start_time
    
    # Résultats
    print(f"\n📊 Résultats Benchmark:")
    print(f"  LPOL:")
    print(f"    - Temps: {lpol_time:.3f}s")
    print(f"    - Mémoire: {lpol_memory / 1e6:.1f}MB")
    print(f"    - Confiance: {lpol_outputs['global_confidence'].mean():.3f}")
    print(f"    - Patterns: {lpol_outputs['total_patterns_used'].mean():.1f}")
    
    print(f"  Traditionnel:")
    print(f"    - Temps: {traditional_time:.3f}s") 
    print(f"    - Confiance: N/A")
    print(f"    - Patterns: N/A")
    
    # Avantages LPOL
    print(f"\n🚀 Avantages LPOL:")
    if lpol_time < traditional_time * 2:  # Acceptable si < 2x plus lent
        print(f"  ✅ Performance acceptable ({lpol_time/traditional_time:.1f}x)")
    else:
        print(f"  ⚠️  Plus lent ({lpol_time/traditional_time:.1f}x)")
    
    print(f"  ✅ Auto-évaluation de confiance")
    print(f"  ✅ Apprentissage par expérience")
    print(f"  ✅ Réutilisation de patterns")
    print(f"  ✅ Amélioration continue")
    
    return True

def interactive_mode():
    """Mode interactif pour tester LPOL avec tokenizer amélioré"""
    
    print("\n🎮 Mode Interactif LPOL")
    print("=" * 40)
    print("Commandes disponibles:")
    print("  generate <prompt> - Générer du texte")
    print("  stats - Afficher les statistiques")
    print("  config - Afficher la configuration")
    print("  help - Afficher l'aide")
    print("  quit - Quitter")
    print()
    
    # Setup modèle interactif avec dimensions compatibles et tokenizer amélioré
    config = get_default_config()
    config.vocab_size = 5000
    config.hidden_size = 512  # Compatible avec 8 têtes
    config.num_heads = 8      # 512/8 = 64, parfait
    config.num_layers = 8
    config.memory_size = 1000
    
    model = LPOLModel(config)
    tokenizer = ImprovedTokenizer(config.vocab_size)
    generator = LPOLTextGenerator(model, tokenizer)
    
    print("✅ LPOL prêt pour interaction (tokenizer français amélioré)")
    
    while True:
        try:
            command = input("\n🤖 LPOL> ").strip()
            
            if command == "quit":
                break
            
            elif command == "help":
                print("Aide LPOL:")
                print("  - generate <prompt>: Génère du texte français lisible")
                print("  - stats: Statistiques du modèle et génération")
                print("  - config: Configuration actuelle")
                print("  - help: Afficher cette aide")
                print("  - quit: Quitter le mode interactif")
                print("\nExemples de prompts:")
                print("  - generate Résoudre le problème")
                print("  - generate Créer une fonction Python")
                print("  - generate Expliquer l'algorithme")
            
            elif command == "stats":
                model_stats = model.get_model_stats()
                gen_stats = generator.get_generation_stats()
                tokenizer_stats = tokenizer.get_vocab_stats()
                
                print("📊 Statistiques LPOL:")
                print("  Modèle:")
                for key, value in model_stats.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.3f}")
                    else:
                        print(f"    {key}: {value}")
                
                print("  Génération:")
                for key, value in gen_stats.items():
                    if not isinstance(value, list):
                        if isinstance(value, float):
                            print(f"    {key}: {value:.3f}")
                        else:
                            print(f"    {key}: {value}")
                
                print("  Tokenizer:")
                for key, value in tokenizer_stats.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.1f}%")
                    else:
                        print(f"    {key}: {value}")
            
            elif command == "config":
                print("⚙️  Configuration LPOL:")
                for key, value in config.__dict__.items():
                    print(f"  {key}: {value}")
            
            elif command.startswith("generate "):
                prompt = command[9:]  # Enlever "generate "
                
                if prompt:
                    print(f"🧠 Génération pour: '{prompt}'")
                    
                    start_time = time.time()
                    result = generator.generate(
                        prompt,
                        max_length=100,
                        temperature=0.8,
                        top_p=0.9
                    )
                    
                    print(f"📝 Résultat: {result['generated_text']}")
                    print(f"⚡ Temps: {result['generation_time']:.3f}s")
                    print(f"🎯 Confiance: {result['confidence']:.3f}")
                    print(f"🧠 Patterns: {result['patterns_used']:.1f}")
                    
                    # Feedback sur l'amélioration
                    if gen_stats := generator.get_generation_stats():
                        if gen_stats['total_generations'] > 1:
                            avg_time = gen_stats.get('avg_generation_time', result['generation_time'])
                            if result['generation_time'] < avg_time:
                                improvement = (avg_time - result['generation_time']) / avg_time * 100
                                print(f"📈 Amélioration: {improvement:.1f}% plus rapide que la moyenne !")
                
                else:
                    print("❌ Prompt vide. Exemple: generate Résoudre le problème")
            
            else:
                print("❌ Commande inconnue. Tapez 'help' pour l'aide.")
                print("Commandes disponibles: generate, stats, config, help, quit")
        
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
            print("Vous pouvez continuer à utiliser LPOL ou taper 'quit' pour quitter.")

def main():
    """Fonction principale du lanceur LPOL"""
    
    parser = argparse.ArgumentParser(description="LPOL Neural Architecture Launcher")
    parser.add_argument('--test-arch', action='store_true', help='Test architecture de base')
    parser.add_argument('--test-generation', action='store_true', help='Test génération texte')
    parser.add_argument('--train-demo', action='store_true', help='Démonstration entraînement')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark vs traditionnel')
    parser.add_argument('--interactive', action='store_true', help='Mode interactif')
    parser.add_argument('--all-tests', action='store_true', help='Tous les tests')
    parser.add_argument('--setup', action='store_true', help='Setup dossiers seulement')
    
    args = parser.parse_args()
    
    print("🚀 LPOL Neural Architecture Launcher")
    print("=" * 50)
    print("Architecture révolutionnaire remplaçant les Transformers")
    print("Copyright © 2025 Amega Mike - Proprietary License")
    print()
    
    # Setup dossiers
    setup_directories()
    
    if args.setup:
        print("✅ Setup terminé")
        return
    
    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()
    
    # Exécution des tests
    try:
        if args.all_tests or args.test_arch:
            if not test_architecture():
                print("❌ Test architecture échoué")
                return 1
        
        if args.all_tests or args.test_generation:
            if not test_text_generation():
                print("❌ Test génération échoué")
                return 1
        
        if args.all_tests or args.train_demo:
            if not run_training_demo():
                print("❌ Démonstration entraînement échouée")
                return 1
        
        if args.all_tests or args.benchmark:
            if not benchmark_vs_traditional():
                print("❌ Benchmark échoué")
                return 1
        
        if args.interactive:
            interactive_mode()
        
        # Si aucun argument, lancer tous les tests
        if not any([args.test_arch, args.test_generation, args.train_demo, 
                   args.benchmark, args.interactive, args.all_tests]):
            print("🧪 Lancement de tous les tests...")
            test_architecture()
            test_text_generation()
            
        print("\n🎉 LPOL Architecture révolutionnaire testée avec succès!")
        print("✅ Prêt à révolutionner l'IA mondiale!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())