"""
LPOL Trainer - Entraîneur Révolutionnaire
Entraîne l'architecture LPOL par résolution de problèmes

Copyright © 2025 Amega Mike - Proprietary License
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import time

# Import de notre architecture
from neural.lpol_neural_core import LPOLModel, LPOLConfig, get_default_config

@dataclass
class TrainingConfig:
    """Configuration d'entraînement LPOL"""
    
    # Hyperparamètres
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Entraînement
    num_epochs: int = 10
    warmup_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 500
    
    # LPOL spécifique
    experience_learning_rate: float = 1e-3
    confidence_loss_weight: float = 0.2
    pattern_usage_weight: float = 0.1
    progressive_difficulty: bool = True
    
    # Chemins
    output_dir: str = "models/lpol_checkpoints"
    data_dir: str = "datasets"
    log_dir: str = "logs"
    
    # Monitoring
    use_wandb: bool = True
    project_name: str = "LPOL-Revolutionary-Training"

class ProblemSolutionDataset(Dataset):
    """Dataset de problèmes-solutions pour LPOL"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Charger données problèmes-solutions
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Chargé {len(self.data)} problèmes-solutions")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: "Problème: ... Solution: ..."
        problem_text = item['problem']
        solution_text = item['solution']
        
        # Tokenization
        full_text = f"Problème: {problem_text}\nSolution: {solution_text}"
        
        tokens = self.tokenizer.encode(full_text, max_length=self.max_length, truncation=True)
        
        # Séparation problème/solution
        separator_idx = tokens.index(self.tokenizer.encode("Solution:")[0]) if self.tokenizer.encode("Solution:")[0] in tokens else len(tokens)//2
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'problem_length': separator_idx,
            'difficulty': item.get('difficulty', 5),
            'category': item.get('category', 'general'),
            'success_score': torch.tensor(item.get('success_score', 1.0), dtype=torch.float32)
        }

class LPOLTrainer:
    """Entraîneur principal pour l'architecture LPOL"""
    
    def __init__(self, model_config: LPOLConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        # Modèle LPOL
        self.model = LPOLModel(model_config)
        
        # Optimiseurs
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=training_config.learning_rate,
            total_steps=10000,  # Sera ajusté
            pct_start=0.1
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Métriques
        self.training_stats = {
            'losses': [],
            'confidences': [],
            'pattern_usage': [],
            'learning_improvements': []
        }
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"LPOL Trainer initialisé sur {self.device}")
        print(f"Paramètres modèle: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Monitoring
        if training_config.use_wandb:
            wandb.init(
                project=training_config.project_name,
                config={**model_config.__dict__, **training_config.__dict__}
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Un pas d'entraînement LPOL"""
        
        self.model.train()
        
        # Données vers device avec conversion de type
        input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
        problem_lengths = batch['problem_length']
        success_scores = batch['success_score'].to(self.device, dtype=torch.float32)
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Calcul des losses
        losses = self._compute_losses(outputs, input_ids, problem_lengths, success_scores)
        
        total_loss = losses['lm_loss'] + losses['confidence_loss'] + losses['pattern_loss']
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
        
        # Optimisation
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Apprentissage LPOL avec feedback
        self._lpol_learning_step(input_ids, success_scores, outputs)
        
        return {
            'total_loss': total_loss.item(),
            'lm_loss': losses['lm_loss'].item(),
            'confidence_loss': losses['confidence_loss'].item(),
            'pattern_loss': losses['pattern_loss'].item(),
            'global_confidence': outputs['global_confidence'].mean().item(),
            'patterns_used': outputs['total_patterns_used'].item()
        }
    
    def _compute_losses(self, outputs: Dict, input_ids: torch.Tensor, 
                       problem_lengths: List[int], success_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calcule les différentes losses LPOL"""
        
        logits = outputs['logits']
        batch_size, seq_len, vocab_size = logits.shape
        
        # 1. Language Modeling Loss classique
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        lm_loss = self.criterion(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        
        # 2. Confidence Loss - LPOL spécifique
        predicted_confidence = outputs['global_confidence'].squeeze(-1)  # Enlever dimension finale si présente
        target_confidence = success_scores.view(-1)  # S'assurer que c'est 1D
        
        # S'assurer que les tensors ont la même forme
        if predicted_confidence.dim() != target_confidence.dim():
            if predicted_confidence.dim() > target_confidence.dim():
                predicted_confidence = predicted_confidence.squeeze()
            else:
                target_confidence = target_confidence.unsqueeze(-1)
        
        confidence_loss = nn.MSELoss()(predicted_confidence.float(), target_confidence.float())
        confidence_loss *= self.training_config.confidence_loss_weight
        
        # 3. Pattern Usage Loss - Encourage l'utilisation de patterns
        pattern_usage = outputs['total_patterns_used']
        target_usage = torch.ones_like(pattern_usage, dtype=torch.float32) * 3.0  # Cible: utiliser ~3 patterns
        
        pattern_loss = nn.MSELoss()(pattern_usage.float(), target_usage.float())
        pattern_loss *= self.training_config.pattern_usage_weight
        
        return {
            'lm_loss': lm_loss,
            'confidence_loss': confidence_loss,
            'pattern_loss': pattern_loss
        }
    
    def _lpol_learning_step(self, input_ids: torch.Tensor, success_scores: torch.Tensor, 
                           outputs: Dict):
        """Étape d'apprentissage LPOL par expérience"""
        
        # Seulement pour les exemples avec succès élevé
        high_success_mask = success_scores > 0.7
        
        if high_success_mask.any():
            # Apprentissage des patterns réussis
            self.model.learn_from_feedback(
                input_ids=input_ids[high_success_mask].to(torch.long),
                target_ids=input_ids[high_success_mask].to(torch.long),  # Auto-apprentissage
                success_scores=success_scores[high_success_mask].to(torch.float32)
            )
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Évaluation du modèle LPOL"""
        
        self.model.eval()
        eval_stats = {
            'eval_loss': 0.0,
            'eval_confidence': 0.0,
            'eval_patterns': 0.0,
            'num_batches': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Évaluation"):
                input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
                success_scores = batch['success_score'].to(self.device, dtype=torch.float32)
                
                outputs = self.model(input_ids)
                losses = self._compute_losses(outputs, input_ids, batch['problem_length'], success_scores)
                
                total_loss = losses['lm_loss'] + losses['confidence_loss'] + losses['pattern_loss']
                
                eval_stats['eval_loss'] += total_loss.item()
                eval_stats['eval_confidence'] += outputs['global_confidence'].mean().item()
                eval_stats['eval_patterns'] += outputs['total_patterns_used'].item()
                eval_stats['num_batches'] += 1
        
        # Moyennes
        for key in eval_stats:
            if key != 'num_batches':
                eval_stats[key] /= eval_stats['num_batches']
        
        return eval_stats
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None):
        """Boucle d'entraînement principale"""
        
        print("🚀 Début entraînement LPOL révolutionnaire...")
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.training_config.num_epochs):
            print(f"\n📚 Époque {epoch+1}/{self.training_config.num_epochs}")
            
            # Barre de progression
            progress_bar = tqdm(train_dataloader, desc=f"Époque {epoch+1}")
            
            epoch_stats = []
            
            for batch in progress_bar:
                # Pas d'entraînement
                step_stats = self.train_step(batch)
                epoch_stats.append(step_stats)
                global_step += 1
                
                # Mise à jour barre de progression
                progress_bar.set_postfix({
                    'Loss': f"{step_stats['total_loss']:.4f}",
                    'Conf': f"{step_stats['global_confidence']:.3f}",
                    'Patterns': f"{step_stats['patterns_used']:.1f}"
                })
                
                # Logging
                if global_step % 100 == 0:
                    self._log_training_stats(step_stats, global_step)
                
                # Évaluation
                if eval_dataloader and global_step % self.training_config.eval_steps == 0:
                    eval_stats = self.evaluate(eval_dataloader)
                    print(f"\n📊 Éval Step {global_step}: Loss={eval_stats['eval_loss']:.4f}")
                    
                    # Sauvegarde si amélioration
                    if eval_stats['eval_loss'] < best_eval_loss:
                        best_eval_loss = eval_stats['eval_loss']
                        self.save_checkpoint(global_step, eval_stats, is_best=True)
                
                # Sauvegarde périodique
                if global_step % self.training_config.save_steps == 0:
                    self.save_checkpoint(global_step, step_stats)
            
            # Stats époque
            epoch_avg = self._average_stats(epoch_stats)
            print(f"📈 Époque {epoch+1} - Loss: {epoch_avg['total_loss']:.4f}, "
                  f"Confidence: {epoch_avg['global_confidence']:.3f}")
            
            # Stats modèle LPOL
            model_stats = self.model.get_model_stats()
            print(f"🧠 Patterns actifs: {model_stats['total_active_patterns']}, "
                  f"Taux succès: {model_stats['average_success_rate']:.3f}")
        
        print("🎉 Entraînement terminé!")
        
        # Sauvegarde finale
        self.save_checkpoint(global_step, epoch_avg, is_final=True)
    
    def _log_training_stats(self, stats: Dict[str, float], step: int):
        """Log des statistiques d'entraînement"""
        
        if self.training_config.use_wandb:
            wandb.log(stats, step=step)
        
        # Historique local
        self.training_stats['losses'].append(stats['total_loss'])
        self.training_stats['confidences'].append(stats['global_confidence'])
        self.training_stats['pattern_usage'].append(stats['patterns_used'])
    
    def _average_stats(self, stats_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Moyenne des statistiques"""
        
        if not stats_list:
            return {}
        
        avg_stats = {}
        for key in stats_list[0].keys():
            avg_stats[key] = sum(stats[key] for stats in stats_list) / len(stats_list)
        
        return avg_stats
    
    def save_checkpoint(self, step: int, stats: Dict[str, float], 
                       is_best: bool = False, is_final: bool = False):
        """Sauvegarde checkpoint"""
        
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # Nom fichier
        if is_final:
            filename = "lpol_final_model.pt"
        elif is_best:
            filename = "lpol_best_model.pt"
        else:
            filename = f"lpol_checkpoint_step_{step}.pt"
        
        filepath = os.path.join(self.training_config.output_dir, filename)
        
        # Données à sauvegarder
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_stats': self.training_stats,
            'current_stats': stats,
            'model_stats': self.model.get_model_stats()
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint sauvé: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        print(f"📂 Checkpoint chargé: {checkpoint_path}")
        print(f"🎯 Step: {checkpoint['step']}")
        print(f"📊 Stats modèle: {checkpoint['model_stats']}")
        
        return checkpoint['step']

def create_sample_dataset(output_path: str, num_samples: int = 1000):
    """Crée un dataset d'exemple pour tester LPOL"""
    
    problems_solutions = []
    
    categories = ['math', 'coding', 'logic', 'text', 'reasoning']
    
    for i in range(num_samples):
        category = np.random.choice(categories)
        difficulty = np.random.randint(1, 6)
        
        if category == 'math':
            a, b = np.random.randint(1, 100, 2)
            problem = f"Calculer {a} + {b}"
            solution = f"La réponse est {a + b}"
            success_score = 1.0
        
        elif category == 'coding':
            problem = "Écrire une fonction qui trouve le maximum d'une liste"
            solution = "def find_max(lst): return max(lst) if lst else None"
            success_score = 0.9
        
        elif category == 'logic':
            problem = "Si A implique B et B implique C, que peut-on dire de A et C?"
            solution = "A implique C (transitivité)"
            success_score = 0.8
        
        else:
            problem = f"Résoudre un problème de {category}"
            solution = f"Solution pour {category}"
            success_score = np.random.uniform(0.6, 1.0)
        
        problems_solutions.append({
            'problem': problem,
            'solution': solution,
            'difficulty': difficulty,
            'category': category,
            'success_score': success_score
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(problems_solutions, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset créé: {output_path} ({num_samples} exemples)")

# Tokenizer simple pour tests
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        # Vocabulaire basique pour tests
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.vocab.update({
            '<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3,
            'Problème:': 4, 'Solution:': 5
        })
        
    def encode(self, text: str, max_length: int = 512, truncation: bool = True) -> List[int]:
        # Tokenization très simple (par mots)
        tokens = text.split()[:max_length] if truncation else text.split()
        
        # Conversion en IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<unk>'])
        
        return token_ids

def collate_fn(batch):
    """Fonction de collation pour gérer le batching"""
    
    # Extraire les éléments
    input_ids = [item['input_ids'] for item in batch]
    problem_lengths = [item['problem_length'] for item in batch]
    success_scores = [item['success_score'] for item in batch]
    
    # Padding des input_ids
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    
    for ids in input_ids:
        if len(ids) < max_len:
            # Padding avec des zéros
            padded = torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)])
        else:
            padded = ids[:max_len]
        padded_input_ids.append(padded)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'problem_length': torch.tensor(problem_lengths, dtype=torch.long),
        'success_score': torch.tensor(success_scores, dtype=torch.float32)
    }

if __name__ == "__main__":
    # Test de l'entraîneur LPOL
    
    print("🧪 Test LPOL Trainer")
    
    # Configuration
    model_config = get_default_config()
    training_config = TrainingConfig(
        batch_size=4,
        num_epochs=2,
        use_wandb=False
    )
    
    # Créer dataset d'exemple
    os.makedirs("datasets", exist_ok=True)
    create_sample_dataset("datasets/sample_problems.json", 100)
    
    # Dataset et dataloader
    tokenizer = SimpleTokenizer()
    dataset = ProblemSolutionDataset("datasets/sample_problems.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Trainer
    trainer = LPOLTrainer(model_config, training_config)
    
    # Entraînement court
    print("Démarrage entraînement test...")
    trainer.train(dataloader)
    
    print("✅ Test terminé!")