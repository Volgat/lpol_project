"""
LPOL Neural Core - Architecture Révolutionnaire
Remplace les Transformers par l'apprentissage basé sur l'expérience model lpol

Copyright © 2025 Amega Mike - Proprietary License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

@dataclass
class LPOLConfig:
    """Configuration de l'architecture LPOL"""
    
    # Dimensions principales
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    # LPOL spécifique
    memory_size: int = 10000          # Taille mémoire patterns
    experience_dim: int = 256         # Dimension expérience
    problem_embedding_dim: int = 128  # Dimension embedding problèmes
    solution_embedding_dim: int = 128 # Dimension embedding solutions
    
    # Apprentissage
    confidence_threshold: float = 0.7  # Seuil confiance
    pattern_match_threshold: float = 0.5  # Seuil matching
    learning_rate_experience: float = 0.01  # LR expérience
    
    # Performance
    max_sequence_length: int = 2048
    dropout_rate: float = 0.1
    use_gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """Validation et ajustements automatiques de la configuration"""
        # S'assurer que hidden_size est divisible par num_heads
        if self.hidden_size % self.num_heads != 0:
            # Ajuster hidden_size au multiple le plus proche
            adjusted_hidden_size = ((self.hidden_size // self.num_heads) + 1) * self.num_heads
            print(f"⚠️  Ajustement hidden_size: {self.hidden_size} -> {adjusted_hidden_size} (divisible par {self.num_heads})")
            self.hidden_size = adjusted_hidden_size
        
        # S'assurer que les dimensions sont cohérentes
        if self.experience_dim > self.hidden_size:
            self.experience_dim = self.hidden_size // 2
            
        if self.problem_embedding_dim > self.hidden_size:
            self.problem_embedding_dim = self.hidden_size // 4
            
        if self.solution_embedding_dim > self.hidden_size:
            self.solution_embedding_dim = self.hidden_size // 4

class ExperienceMemory(nn.Module):
    """Mémoire d'expérience LPOL - Remplace l'attention classique"""
    
    def __init__(self, config: LPOLConfig):
        super().__init__()
        self.config = config
        
        # Banques mémoire
        self.problem_memory = nn.Parameter(
            torch.randn(config.memory_size, config.problem_embedding_dim)
        )
        self.solution_memory = nn.Parameter(
            torch.randn(config.memory_size, config.solution_embedding_dim)
        )
        self.confidence_memory = nn.Parameter(
            torch.randn(config.memory_size, 1)
        )
        
        # Projections
        self.problem_proj = nn.Linear(config.hidden_size, config.problem_embedding_dim)
        self.solution_proj = nn.Linear(config.hidden_size, config.solution_embedding_dim)
        self.output_proj = nn.Linear(config.solution_embedding_dim, config.hidden_size)
        
        # Mécanisme apprentissage
        self.experience_encoder = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.experience_dim),
            nn.ReLU(),
            nn.Linear(config.experience_dim, config.experience_dim),
            nn.Tanh()
        )
        
        # Compteur utilisation patterns
        self.register_buffer('pattern_usage', torch.zeros(config.memory_size))
        self.register_buffer('pattern_success', torch.zeros(config.memory_size))
        
    def forward(self, x: torch.Tensor, problem_context: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass avec récupération expérience
        
        Args:
            x: Tokens d'entrée [batch, seq_len, hidden_size]
            problem_context: Contexte problème [batch, hidden_size]
        
        Returns:
            output: Sortie enrichie par expérience
            experience_info: Infos sur l'expérience utilisée
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 1. Encoder le problème actuel
        if problem_context is None:
            problem_context = x.mean(dim=1)  # Moyenne des tokens
        
        current_problem = self.problem_proj(problem_context)  # [batch, problem_dim]
        
        # 2. Rechercher patterns similaires dans la mémoire
        similarity_scores = torch.matmul(
            current_problem, 
            self.problem_memory.t()
        )  # [batch, memory_size]
        
        # 3. Pondérer par confiance et utilisation passée
        usage_boost = torch.log(self.pattern_usage + 1).unsqueeze(0)  # [1, memory_size]
        success_rate = self.pattern_success / (self.pattern_usage + 1e-8)
        confidence_boost = self.confidence_memory.squeeze(-1).unsqueeze(0)  # [1, memory_size]
        
        # Score final combiné
        final_scores = (
            similarity_scores + 
            0.1 * usage_boost + 
            0.2 * confidence_boost.expand_as(similarity_scores) +
            0.3 * success_rate.unsqueeze(0).expand_as(similarity_scores)
        )
        
        # 4. Sélectionner top patterns
        top_k = min(5, self.config.memory_size)
        top_scores, top_indices = torch.topk(final_scores, top_k, dim=-1)
        
        # 5. Récupérer solutions correspondantes
        # Utiliser gather pour batch processing
        batch_indices = torch.arange(batch_size).unsqueeze(-1).expand(-1, top_k)
        selected_solutions = self.solution_memory[top_indices]  # [batch, top_k, solution_dim]
        
        # 6. Attention pondérée sur les solutions
        attention_weights = F.softmax(top_scores / math.sqrt(self.config.solution_embedding_dim), dim=-1)
        
        # 7. Combiner solutions avec attention
        combined_solution = torch.sum(
            selected_solutions * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch, solution_dim]
        
        # 8. Projeter vers espace caché
        experience_output = self.output_proj(combined_solution)  # [batch, hidden_size]
        
        # 9. Intégrer avec séquence originale
        # Broadcast pour toute la séquence
        experience_broadcast = experience_output.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combinaison adaptative
        gate = torch.sigmoid(
            torch.matmul(x, experience_broadcast.transpose(-2, -1)).mean(dim=-1, keepdim=True)
        )
        
        output = gate * experience_broadcast + (1 - gate) * x
        
        # Informations expérience pour monitoring
        experience_info = {
            'top_patterns': top_indices,
            'pattern_scores': top_scores,
            'attention_weights': attention_weights,
            'confidence': top_scores.max(dim=-1)[0],
            'num_patterns_used': (top_scores > self.config.pattern_match_threshold).sum(dim=-1)
        }
        
        return output, experience_info
    
    def learn_from_feedback(self, problem_context: torch.Tensor, solution: torch.Tensor, 
                          success: torch.Tensor, pattern_indices: torch.Tensor):
        """
        Met à jour la mémoire basée sur le feedback
        
        Args:
            problem_context: Contexte du problème [batch, hidden_size]
            solution: Solution générée [batch, hidden_size] 
            success: Succès (0/1) [batch]
            pattern_indices: Indices patterns utilisés [batch, top_k]
        """
        
        with torch.no_grad():
            # Mettre à jour statistiques d'utilisation
            for b in range(problem_context.shape[0]):
                for idx in pattern_indices[b]:
                    if idx < self.config.memory_size:
                        self.pattern_usage[idx] += 1
                        if success[b] > 0.5:
                            self.pattern_success[idx] += 1
            
            # Si succès, potentiellement ajouter nouveau pattern
            for b in range(problem_context.shape[0]):
                if success[b] > self.config.confidence_threshold:
                    self._add_successful_pattern(
                        self.problem_proj(problem_context[b]),
                        self.solution_proj(solution[b]),
                        success[b]
                    )
    
    def _add_successful_pattern(self, problem_embedding: torch.Tensor, 
                              solution_embedding: torch.Tensor, confidence: float):
        """Ajoute un nouveau pattern réussi à la mémoire"""
        
        # Trouver l'emplacement le moins utilisé
        least_used_idx = self.pattern_usage.argmin().item()
        
        # Remplacer si utilisé moins de 5 fois ou nouvelle confiance > ancienne
        if (self.pattern_usage[least_used_idx] < 5 or 
            confidence > self.confidence_memory[least_used_idx].item()):
            
            self.problem_memory[least_used_idx] = problem_embedding.detach()
            self.solution_memory[least_used_idx] = solution_embedding.detach()
            self.confidence_memory[least_used_idx] = confidence
            self.pattern_usage[least_used_idx] = 1
            self.pattern_success[least_used_idx] = 1

class LPOLAttention(nn.Module):
    """Attention LPOL - Basée sur l'expérience plutôt que sur les mots"""
    
    def __init__(self, config: LPOLConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # Vérification de la cohérence des dimensions
        assert config.hidden_size % config.num_heads == 0, f"hidden_size ({config.hidden_size}) doit être divisible par num_heads ({config.num_heads})"
        
        # Projections classiques
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        # LPOL: Projection expérience
        self.experience_query = nn.Linear(config.hidden_size, config.experience_dim)
        self.experience_key = nn.Linear(config.hidden_size, config.experience_dim)
        
        # Fusion experience + attention
        self.experience_gate = nn.Linear(config.experience_dim * 2, 1)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor, experience_info: Dict = None) -> torch.Tensor:
        """
        Attention enrichie par l'expérience
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            experience_info: Infos expérience du ExperienceMemory
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Attention classique
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scores attention classique
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # LPOL: Modifier attention basée sur expérience
        if experience_info is not None:
            exp_q = self.experience_query(x)  # [batch, seq_len, exp_dim]
            exp_k = self.experience_key(x)    # [batch, seq_len, exp_dim]
            
            # Scores basés sur expérience
            exp_scores = torch.matmul(exp_q, exp_k.transpose(-2, -1)) / math.sqrt(self.config.experience_dim)
            
            # Pondération par confiance des patterns
            if 'confidence' in experience_info:
                confidence_weight = experience_info['confidence'].unsqueeze(-1).unsqueeze(-1)
                exp_scores = exp_scores * confidence_weight
            
            # Fusionner avec attention classique
            # Répéter exp_scores pour chaque tête d'attention
            exp_scores_expanded = exp_scores.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attention_scores = attention_scores + 0.3 * exp_scores_expanded
        
        # Softmax et application
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Application aux values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Projection finale
        output = self.output_proj(context)
        
        return output

class LPOLLayer(nn.Module):
    """Couche LPOL complète - Remplace TransformerLayer"""
    
    def __init__(self, config: LPOLConfig):
        super().__init__()
        self.config = config
        
        # Composants LPOL
        self.experience_memory = ExperienceMemory(config)
        self.attention = LPOLAttention(config)
        
        # Composants classiques améliorés
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        
        # FFN avec boost expérience
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout_rate)
        )
        
        # Mécanisme de confiance
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        
    def forward(self, x: torch.Tensor, problem_context: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass couche LPOL
        """
        # 1. Récupération expérience
        experience_output, experience_info = self.experience_memory(x, problem_context)
        
        # 2. Attention enrichie par expérience
        attention_output = self.attention(experience_output, experience_info)
        
        # 3. Résiduelle + norm
        x = self.layer_norm1(x + attention_output)
        
        # 4. FFN
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # 5. Calcul confiance
        confidence = torch.sigmoid(self.confidence_head(x.mean(dim=1)))
        experience_info['layer_confidence'] = confidence
        
        return x, experience_info

class LPOLModel(nn.Module):
    """Modèle LPOL complet - Remplace GPT/BERT/T5"""
    
    def __init__(self, config: LPOLConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        
        # Couches LPOL
        self.layers = nn.ModuleList([
            LPOLLayer(config) for _ in range(config.num_layers)
        ])
        
        # Heads de sortie
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Apprentissage global
        self.global_experience_tracker = nn.Parameter(torch.zeros(1))
        
        self.init_weights()
    
    def init_weights(self):
        """Initialisation des poids"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, problem_context: torch.Tensor = None) -> Dict[str, Any]:
        """
        Forward pass modèle LPOL complet
        
        Args:
            input_ids: Tokens d'entrée [batch, seq_len]
            problem_context: Contexte problème optionnel
        
        Returns:
            Dictionnaire avec logits, expérience, confiance
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        # Passage dans les couches LPOL
        all_experience_info = []
        for layer in self.layers:
            x, experience_info = layer(x, problem_context)
            all_experience_info.append(experience_info)
        
        # Normalisation finale
        x = self.layer_norm(x)
        
        # Génération logits
        logits = self.lm_head(x)
        
        # Agrégation expérience globale
        global_confidence = torch.stack([
            info['layer_confidence'] for info in all_experience_info
        ]).mean(dim=0)
        
        total_patterns_used = sum([
            info['num_patterns_used'].float().mean() 
            for info in all_experience_info
        ]) / len(all_experience_info)
        
        return {
            'logits': logits,
            'global_confidence': global_confidence,
            'total_patterns_used': total_patterns_used,
            'layer_experiences': all_experience_info
        }
    
    def learn_from_feedback(self, input_ids: torch.Tensor, target_ids: torch.Tensor, 
                          success_scores: torch.Tensor):
        """
        Apprentissage à partir du feedback sur génération
        
        Args:
            input_ids: Input [batch, seq_len]
            target_ids: Target généré [batch, seq_len] 
            success_scores: Scores de réussite [batch]
        """
        
        # Forward pour récupérer infos expérience
        with torch.no_grad():
            outputs = self.forward(input_ids)
            
            # Mettre à jour chaque couche
            for i, layer in enumerate(self.layers):
                experience_info = outputs['layer_experiences'][i]
                problem_context = self.token_embedding(input_ids).mean(dim=1)
                solution_context = self.token_embedding(target_ids).mean(dim=1)
                
                layer.experience_memory.learn_from_feedback(
                    problem_context=problem_context,
                    solution=solution_context,
                    success=success_scores,
                    pattern_indices=experience_info['top_patterns']
                )
            
            # Mettre à jour tracker global
            self.global_experience_tracker += success_scores.mean()

    def get_model_stats(self) -> Dict[str, Any]:
        """Statistiques du modèle LPOL"""
        
        total_patterns = 0
        total_usage = 0
        total_success_rate = 0
        
        for layer in self.layers:
            memory = layer.experience_memory
            total_patterns += (memory.pattern_usage > 0).sum().item()
            total_usage += memory.pattern_usage.sum().item()
            
            success_rate = (memory.pattern_success / (memory.pattern_usage + 1e-8)).mean().item()
            total_success_rate += success_rate
        
        return {
            'total_active_patterns': total_patterns,
            'total_pattern_usage': total_usage,
            'average_success_rate': total_success_rate / len(self.layers),
            'global_experience': self.global_experience_tracker.item(),
            'memory_efficiency': total_patterns / (len(self.layers) * self.config.memory_size)
        }

# Configuration par défaut optimisée
def get_default_config() -> LPOLConfig:
    """Configuration LPOL par défaut optimisée"""
    return LPOLConfig(
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        memory_size=5000,
        experience_dim=256,
        problem_embedding_dim=128,
        solution_embedding_dim=128,
        confidence_threshold=0.7,
        pattern_match_threshold=0.5,
        max_sequence_length=2048,
        dropout_rate=0.1
    )