"""
Tokenizer Amélioré pour LPOL avec vocabulaire français étendu
"""

class LPOLTokenizerImproved:
    """Tokenizer amélioré pour génération lisible"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        
        # Tokens spéciaux
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
            '<problem>': 4,
            '<solution>': 5
        }
        
        # Vocabulaire français étendu
        self.french_vocab = [
            # Mots courants
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout',
            'plus', 'par', 'grand', 'premier', 'même', 'nouveau', 'bon', 'autre',
            'faire', 'savoir', 'voir', 'donner', 'homme', 'jour', 'temps', 'main',
            
            # Mots techniques/problèmes
            'problème', 'solution', 'réponse', 'question', 'résoudre', 'calculer',
            'fonction', 'algorithme', 'code', 'python', 'programme', 'variable',
            'résultat', 'calcul', 'nombre', 'liste', 'texte', 'données', 'fichier',
            'classe', 'méthode', 'objet', 'import', 'module', 'bibliothèque',
            
            # Verbes d'action
            'créer', 'générer', 'implémenter', 'optimiser', 'tester', 'valider',
            'exécuter', 'compiler', 'debugger', 'analyser', 'traiter', 'transformer',
            'construire', 'développer', 'concevoir', 'planifier', 'organiser',
            
            # Mots logiques
            'si', 'alors', 'sinon', 'pour', 'tant', 'que', 'while', 'do', 'return',
            'true', 'false', 'null', 'void', 'int', 'string', 'float', 'bool',
            'array', 'object', 'dict', 'set', 'tuple', 'range', 'len', 'max', 'min',
            
            # Connecteurs
            'donc', 'ainsi', 'par', 'conséquent', 'cependant', 'néanmoins', 'mais',
            'ou', 'car', 'parce', 'afin', 'selon', 'grâce', 'malgré', 'pendant',
            
            # Nombres et quantités
            'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'dix',
            'premier', 'deuxième', 'dernier', 'plusieurs', 'beaucoup', 'peu', 'assez',
            
            # Qualificatifs
            'bon', 'meilleur', 'excellent', 'parfait', 'optimal', 'efficace', 'rapide',
            'simple', 'complexe', 'difficile', 'facile', 'important', 'nécessaire',
            'possible', 'impossible', 'correct', 'incorrect', 'valide', 'invalide'
        ]
        
        # Construction vocabulaire complet
        self.vocab = self.special_tokens.copy()
        
        # Ajouter vocabulaire français
        for i, word in enumerate(self.french_vocab):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Compléter avec tokens génériques si nécessaire
        current_size = len(self.vocab)
        if current_size < vocab_size:
            for i in range(current_size, vocab_size):
                self.vocab[f'word_{i}'] = i
        
        # Dictionnaire inverse
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Tokenizer amélioré: {len(self.vocab)} tokens, dont {len(self.french_vocab)} mots français")
    
    def encode(self, text: str, max_length: int = 512, truncation: bool = True) -> list:
        """Encode le texte en IDs"""
        
        # Nettoyage et tokenisation
        text = text.lower().strip()
        words = text.replace(',', ' ,').replace('.', ' .').replace(':', ' :').split()
        
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
        """Décode les IDs en texte lisible"""
        
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Ignorer tokens spéciaux dans la sortie
                if not token.startswith('<') and not token.startswith('word_'):
                    tokens.append(token)
        
        # Reconstitution avec espaces appropriés
        text = ' '.join(tokens)
        
        # Nettoyage ponctuation
        text = text.replace(' ,', ',').replace(' .', '.').replace(' :', ':')
        text = text.replace(' !', '!').replace(' ?', '?').replace(' ;', ';')
        
        return text.strip()
    
    def get_vocab_stats(self):
        """Statistiques du vocabulaire"""
        return {
            'total_tokens': len(self.vocab),
            'french_words': len(self.french_vocab),
            'special_tokens': len(self.special_tokens),
            'coverage': len(self.french_vocab) / len(self.vocab) * 100
        }

# Remplacez le tokenizer dans text_generation.py
def get_improved_tokenizer(vocab_size: int = 5000):
    """Retourne le tokenizer amélioré"""
    return LPOLTokenizerImproved(vocab_size)