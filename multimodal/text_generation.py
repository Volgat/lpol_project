"""
LPOL Text Generation - Génération de Texte Révolutionnaire
Utilise l'architecture LPOL pour la génération basée sur l'expérience

Copyright © 2025 Amega Mike - Proprietary License
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import time
from dataclasses import dataclass

# Imports LPOL
from neural.lpol_neural_core import LPOLModel, LPOLConfig, get_default_config

@dataclass
class GenerationConfig:
    """Configuration pour la génération LPOL"""
    
    # Paramètres génération
    max_length: int = 512
    min_length: int = 10
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # LPOL spécifique
    confidence_threshold: float = 0.7
    experience_weight: float = 0.3
    adaptive_temperature: bool = True
    pattern_guidance: bool = True
    
    # Contrôle génération
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    
    # Tokens spéciaux
    pad_token_id: int = 0
    eos_token_id: int = 3
    bos_token_id: int = 2

class LPOLTextGenerator:
    """Générateur de texte basé sur l'architecture LPOL"""
    
    def __init__(self, model: LPOLModel, tokenizer, config: GenerationConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        self.device = next(model.parameters()).device
        
        # Statistiques génération
        self.generation_stats = {
            'total_generations': 0,
            'avg_confidence': 0.0,
            'patterns_usage': [],
            'generation_times': []
        }
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Génère du texte à partir d'un prompt avec LPOL
        
        Args:
            prompt: Texte de départ
            **kwargs: Paramètres de génération
        
        Returns:
            Dictionnaire avec texte généré et métadonnées LPOL
        """
        
        # Mise à jour config avec kwargs
        gen_config = self._update_config(kwargs)
        
        # Tokenisation du prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, truncation=True),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        # Génération avec LPOL
        start_time = time.time()
        
        with torch.no_grad():
            if gen_config.num_beams > 1:
                generated_ids, metadata = self._beam_search_generate(input_ids, gen_config)
            else:
                generated_ids, metadata = self._sampling_generate(input_ids, gen_config)
        
        generation_time = time.time() - start_time
        
        # Décodage
        generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())
        
        # Mise à jour statistiques
        self._update_stats(metadata, generation_time)
        
        return {
            'generated_text': generated_text,
            'prompt': prompt,
            'generation_time': generation_time,
            'lpol_metadata': metadata,
            'confidence': metadata.get('avg_confidence', 0.0),
            'patterns_used': metadata.get('total_patterns_used', 0),
            'tokens_generated': generated_ids.shape[1] - input_ids.shape[1]
        }
    
    def _sampling_generate(self, input_ids: torch.Tensor, 
                          config: GenerationConfig) -> Tuple[torch.Tensor, Dict]:
        """Génération par échantillonnage avec guidage LPOL"""
        
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # Accumulateurs métadonnées LPOL
        all_confidences = []
        all_patterns_used = []
        generation_metadata = []
        
        # Cache pour patterns réutilisables
        pattern_cache = {}
        
        while current_length < config.max_length:
            # Forward pass LPOL
            outputs = self.model(input_ids)
            
            # Extraction métadonnées LPOL
            confidence = outputs['global_confidence'].item()
            patterns_used = outputs['total_patterns_used'].item()
            
            all_confidences.append(confidence)
            all_patterns_used.append(patterns_used)
            
            # Logits pour le prochain token
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Ajustement température basé sur confiance LPOL
            if config.adaptive_temperature:
                # Plus la confiance est élevée, plus la température est basse
                adaptive_temp = config.temperature * (2.0 - confidence)
                next_token_logits = next_token_logits / adaptive_temp
            else:
                next_token_logits = next_token_logits / config.temperature
            
            # Guidage par patterns LPOL
            if config.pattern_guidance and patterns_used > 0:
                next_token_logits = self._apply_pattern_guidance(
                    next_token_logits, outputs, pattern_cache
                )
            
            # Pénalité répétition
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, input_ids, config.repetition_penalty
                )
            
            # Filtrage top-k et top-p
            if config.top_k > 0:
                next_token_logits = self._top_k_filter(next_token_logits, config.top_k)
            
            if config.top_p < 1.0:
                next_token_logits = self._top_p_filter(next_token_logits, config.top_p)
            
            # Échantillonnage
            probs = F.softmax(next_token_logits, dim=-1)
            
            if config.do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Ajout du token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            current_length += 1
            
            # Arrêt si EOS
            if next_token.item() == config.eos_token_id:
                break
            
            # Arrêt si confiance trop faible (LPOL spécifique)
            if confidence < config.confidence_threshold * 0.3:
                generation_metadata.append({
                    'step': current_length,
                    'reason': 'low_confidence',
                    'confidence': confidence
                })
                break
        
        # Métadonnées finales
        metadata = {
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0.0,
            'total_patterns_used': np.mean(all_patterns_used) if all_patterns_used else 0.0,
            'confidence_trajectory': all_confidences,
            'patterns_trajectory': all_patterns_used,
            'generation_steps': len(all_confidences),
            'early_stopping_reason': generation_metadata[-1] if generation_metadata else None
        }
        
        return input_ids, metadata
    
    def _beam_search_generate(self, input_ids: torch.Tensor, 
                             config: GenerationConfig) -> Tuple[torch.Tensor, Dict]:
        """Génération par beam search avec guidage LPOL"""
        
        # Pour simplicité, utilise sampling pour l'instant
        # TODO: Implémenter beam search avec guidage LPOL
        return self._sampling_generate(input_ids, config)
    
    def _apply_pattern_guidance(self, logits: torch.Tensor, outputs: Dict, 
                               pattern_cache: Dict) -> torch.Tensor:
        """Applique le guidage basé sur les patterns LPOL"""
        
        # Récupération des patterns actifs
        layer_experiences = outputs.get('layer_experiences', [])
        
        if not layer_experiences:
            return logits
        
        # Accumuler les influences des patterns
        pattern_influence = torch.zeros_like(logits)
        
        for layer_exp in layer_experiences:
            if 'attention_weights' in layer_exp:
                # Utiliser les poids d'attention comme guidage
                attention_weights = layer_exp['attention_weights']
                
                # Boost probabilités basé sur attention patterns
                # Simplification: boost tokens fréquents dans patterns réussis
                pattern_influence += 0.1 * attention_weights.mean() * torch.ones_like(logits)
        
        return logits + pattern_influence
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """Applique la pénalité de répétition"""
        
        for token_id in set(input_ids[0].tolist()):
            logits[0, token_id] /= penalty
        
        return logits
    
    def _top_k_filter(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filtre top-k"""
        
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1, None]
        
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filtre top-p (nucleus sampling)"""
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Retirer tokens avec probabilité cumulative > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def _update_config(self, kwargs: Dict) -> GenerationConfig:
        """Met à jour la configuration avec les arguments"""
        
        config = GenerationConfig(**{**self.config.__dict__, **kwargs})
        return config
    
    def _update_stats(self, metadata: Dict, generation_time: float):
        """Met à jour les statistiques de génération"""
        
        self.generation_stats['total_generations'] += 1
        self.generation_stats['generation_times'].append(generation_time)
        
        if 'avg_confidence' in metadata:
            # Moyenne mobile de la confiance
            alpha = 0.1
            self.generation_stats['avg_confidence'] = (
                alpha * metadata['avg_confidence'] + 
                (1 - alpha) * self.generation_stats['avg_confidence']
            )
        
        if 'total_patterns_used' in metadata:
            self.generation_stats['patterns_usage'].append(metadata['total_patterns_used'])
    
    def interactive_generation(self):
        """Mode génération interactive pour tester LPOL"""
        
        print("🤖 LPOL Interactive Text Generation")
        print("=" * 40)
        print("Tapez 'quit' pour quitter")
        print()
        
        while True:
            prompt = input("💬 Prompt: ")
            
            if prompt.lower() == 'quit':
                break
            
            print("\n🧠 LPOL en train de générer...")
            
            result = self.generate(
                prompt,
                max_length=200,
                temperature=0.8,
                top_p=0.9
            )
            
            print(f"\n📝 Généré: {result['generated_text']}")
            print(f"⚡ Temps: {result['generation_time']:.2f}s")
            print(f"🎯 Confiance: {result['confidence']:.3f}")
            print(f"🧠 Patterns: {result['patterns_used']:.1f}")
            print("-" * 40)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Génération en batch pour plusieurs prompts"""
        
        results = []
        
        print(f"🚀 Génération batch de {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            print(f"Génération {i+1}/{len(prompts)}")
            result = self.generate(prompt, **kwargs)
            results.append(result)
        
        return results
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de génération"""
        
        stats = self.generation_stats.copy()
        
        if stats['generation_times']:
            stats['avg_generation_time'] = np.mean(stats['generation_times'])
            stats['total_generation_time'] = np.sum(stats['generation_times'])
        
        if stats['patterns_usage']:
            stats['avg_patterns_per_generation'] = np.mean(stats['patterns_usage'])
        
        return stats

# Tokenizer simple amélioré pour tests
class LPOLTokenizer:
    """Tokenizer simple pour LPOL avec support des tokens spéciaux"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        
        # Tokens spéciaux
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
            '<problem>': 4,
            '<solution>': 5,
            '<confidence>': 6
        }
        
        # Vocabulaire de base (mots courants français)
        self.base_vocab = [
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir',
            'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne',
            'se', 'pas', 'tout', 'plus', 'par', 'grand', 'premier', 'même',
            'problème', 'solution', 'réponse', 'question', 'code', 'fonction',
            'variable', 'résultat', 'calcul', 'nombre', 'liste', 'texte'
        ]
        
        # Construction vocabulaire complet
        self.vocab = self.special_tokens.copy()
        
        for i, word in enumerate(self.base_vocab):
            self.vocab[word] = len(self.special_tokens) + i
        
        # Compléter avec tokens génériques
        for i in range(len(self.vocab), vocab_size):
            self.vocab[f'token_{i}'] = i
        
        # Dictionnaire inverse
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, max_length: int = 512, truncation: bool = True) -> List[int]:
        """Encode le texte en IDs"""
        
        # Tokenisation simple par mots
        words = text.lower().split()
        
        if truncation:
            words = words[:max_length-2]  # Place pour BOS/EOS
        
        # Conversion en IDs
        token_ids = [self.vocab['<bos>']]
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab['<unk>'])
        
        token_ids.append(self.vocab['<eos>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Décode les IDs en texte"""
        
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Ignorer tokens spéciaux dans la sortie
                if not token.startswith('<'):
                    tokens.append(token)
        
        return ' '.join(tokens)

def demo_lpol_text_generation():
    """Démonstration de la génération de texte LPOL"""
    
    print("🚀 Démonstration Génération Texte LPOL")
    print("=" * 50)
    
    # Configuration
    model_config = get_default_config()
    model_config.vocab_size = 10000  # Plus petit pour le test
    
    # Modèle LPOL
    model = LPOLModel(model_config)
    tokenizer = LPOLTokenizer(model_config.vocab_size)
    
    # Générateur
    generator = LPOLTextGenerator(model, tokenizer)
    
    # Prompts de test
    test_prompts = [
        "Résoudre le problème:",
        "Écrire une fonction qui",
        "La solution est",
        "Pour calculer"
    ]
    
    print("\n🧪 Tests de génération:")
    print("-" * 30)
    
    for prompt in test_prompts:
        print(f"\n💬 Prompt: '{prompt}'")
        
        result = generator.generate(
            prompt,
            max_length=100,
            temperature=0.8
        )
        
        print(f"📝 Généré: {result['generated_text']}")
        print(f"⚡ Temps: {result['generation_time']:.3f}s")
        print(f"🎯 Confiance: {result['confidence']:.3f}")
        print(f"🧠 Patterns: {result['patterns_used']:.1f}")
    
    # Statistiques
    print(f"\n📊 Statistiques génération:")
    stats = generator.get_generation_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    demo_lpol_text_generation()