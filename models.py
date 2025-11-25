# gd_cdss/models.py

"""
Neural network model definitions for gene-drug association prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union


class DualBranchNet(nn.Module):
    """
    Dual-branch neural network for gene-drug association prediction.
    
    Architecture:
        - Drug encoder: processes drug features
        - Protein encoder: processes protein features  
        - Fusion layer: combines encoded features and predicts association
    """
    
    def __init__(self, d_in: int, p_in: int, hidden: int = 128):
        """
        Initialize the dual-branch network.
        
        Args:
            d_in: Drug feature dimension
            p_in: Protein feature dimension
            hidden: Hidden layer size
        """
        super().__init__()
        
        # Drug encoder branch
        self.drug_encoder = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )
        
        # Protein encoder branch
        self.prot_encoder = nn.Sequential(
            nn.Linear(p_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, drug: torch.Tensor, prot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            drug: Drug feature tensor [batch_size, d_in]
            prot: Protein feature tensor [batch_size, p_in]
            
        Returns:
            Predicted association probability [batch_size]
        """
        # Encode drug and protein features
        d_encoded = self.drug_encoder(drug)
        p_encoded = self.prot_encoder(prot)
        
        # Concatenate and fuse
        fused = torch.cat([d_encoded, p_encoded], dim=1)
        
        # Predict association probability
        return self.fusion(fused).squeeze(1)
    
    def predict_proba(self, 
                     drug_features: Union[np.ndarray, torch.Tensor],
                     prot_features: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Predict association probability for a single drug-protein pair.
        
        Args:
            drug_features: Drug feature vector
            prot_features: Protein feature vector
            
        Returns:
            Association probability (0-1)
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensors if needed
            if isinstance(drug_features, np.ndarray):
                drug_features = torch.tensor(drug_features, dtype=torch.float32)
            if isinstance(prot_features, np.ndarray):
                prot_features = torch.tensor(prot_features, dtype=torch.float32)
            
            # Ensure batch dimension
            if drug_features.dim() == 1:
                drug_features = drug_features.unsqueeze(0)
            if prot_features.dim() == 1:
                prot_features = prot_features.unsqueeze(0)
            
            # Predict
            prob = self(drug_features, prot_features)
            
            return float(prob.item())
    
    def predict_batch(self,
                     drug_features: Union[np.ndarray, torch.Tensor],
                     prot_features: Union[np.ndarray, torch.Tensor],
                     batch_size: int = 64) -> np.ndarray:
        """
        Predict association probabilities for multiple pairs.
        
        Args:
            drug_features: Drug feature matrix [n_samples, d_in]
            prot_features: Protein feature matrix [n_samples, p_in]
            batch_size: Batch size for processing
            
        Returns:
            Array of association probabilities [n_samples]
        """
        self.eval()
        
        # Convert to tensors if needed
        if isinstance(drug_features, np.ndarray):
            drug_features = torch.tensor(drug_features, dtype=torch.float32)
        if isinstance(prot_features, np.ndarray):
            prot_features = torch.tensor(prot_features, dtype=torch.float32)
        
        n_samples = drug_features.shape[0]
        predictions = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                drug_batch = drug_features[i:i+batch_size]
                prot_batch = prot_features[i:i+batch_size]
                
                probs = self(drug_batch, prot_batch)
                predictions.extend(probs.cpu().numpy())
        
        return np.array(predictions)
    
    def encode_drug(self, drug_features: torch.Tensor) -> torch.Tensor:
        """
        Encode drug features to latent representation.
        
        Args:
            drug_features: Drug feature tensor
            
        Returns:
            Encoded drug representation
        """
        return self.drug_encoder(drug_features)
    
    def encode_protein(self, prot_features: torch.Tensor) -> torch.Tensor:
        """
        Encode protein features to latent representation.
        
        Args:
            prot_features: Protein feature tensor
            
        Returns:
            Encoded protein representation
        """
        return self.prot_encoder(prot_features)


class EnsembleModel:
    """
    Ensemble of multiple DualBranchNet models for improved predictions.
    """
    
    def __init__(self, models: list):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained DualBranchNet models
        """
        self.models = models
        
        for model in self.models:
            model.eval()
    
    def predict_proba(self,
                     drug_features: Union[np.ndarray, torch.Tensor],
                     prot_features: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Predict using ensemble averaging.
        
        Args:
            drug_features: Drug feature vector
            prot_features: Protein feature vector
            
        Returns:
            Ensemble-averaged association probability
        """
        predictions = []
        
        for model in self.models:
            prob = model.predict_proba(drug_features, prot_features)
            predictions.append(prob)
        
        return float(np.mean(predictions))
    
    def predict_batch(self,
                     drug_features: Union[np.ndarray, torch.Tensor],
                     prot_features: Union[np.ndarray, torch.Tensor],
                     batch_size: int = 64) -> np.ndarray:
        """
        Predict batch using ensemble averaging.
        
        Args:
            drug_features: Drug feature matrix
            prot_features: Protein feature matrix
            batch_size: Batch size for processing
            
        Returns:
            Array of ensemble-averaged probabilities
        """
        all_predictions = []
        
        for model in self.models:
            preds = model.predict_batch(drug_features, prot_features, batch_size)
            all_predictions.append(preds)
        
        # Average across models
        return np.mean(all_predictions, axis=0)
