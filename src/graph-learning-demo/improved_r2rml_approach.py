#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import pandas as pd
from rdflib import Graph, URIRef
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU, Dropout, MultiheadAttention, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from datetime import datetime

# Try to import wandb, but don't fail if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Will log metrics to JSON file instead.")

class MetricsLogger:
    def __init__(self, use_wandb: bool = False, project_name: str = "clinical-trials-gnn"):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name, name="improved-r2rml-approach")
        else:
            self.metrics = []
            self.log_dir = "logs"
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(
                self.log_dir, 
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
    
    def log(self, metrics: Dict[str, float]) -> None:
        if self.use_wandb:
            wandb.log(metrics)
        else:
            self.metrics.append({
                "step": len(self.metrics),
                "timestamp": datetime.now().isoformat(),
                **metrics
            })
            # Save to file after each update
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
    
    def finish(self) -> None:
        if self.use_wandb:
            wandb.finish()

class ImprovedFeatureEncoder(torch.nn.Module):
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.feature_encoders = torch.nn.ModuleDict({
            'numerical': torch.nn.Sequential(
                Linear(1, hidden_dim),
                LayerNorm(hidden_dim),
                ReLU()
            ),
            'categorical': torch.nn.Embedding(num_embeddings=1000, embedding_dim=hidden_dim),
            'temporal': LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        })
        
        self.feature_attention = MultiheadAttention(hidden_dim, num_heads=4)
        
    def forward(self, features: Dict[str, torch.Tensor], feature_types: Dict[str, str]) -> torch.Tensor:
        encoded_features = []
        for feat_name, feat in features.items():
            feat_type = feature_types[feat_name]
            if feat_type == 'temporal':
                encoded = self.feature_encoders[feat_type](feat.unsqueeze(-1))[0]
            elif feat_type == 'categorical':
                encoded = self.feature_encoders[feat_type](feat)
            else:
                encoded = self.feature_encoders[feat_type](feat.unsqueeze(-1))
            encoded_features.append(encoded)
            
        features_stack = torch.stack(encoded_features, dim=1)
        attended_features, _ = self.feature_attention(
            features_stack, features_stack, features_stack
        )
        return attended_features

class ImprovedTemporalEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.temporal_conv = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_attention = MultiheadAttention(hidden_dim, num_heads=4)
        self.time_embedding = Linear(1, hidden_dim)
        
    def forward(self, x: torch.Tensor, time_values: torch.Tensor) -> torch.Tensor:
        # Encode absolute time
        time_emb = self.time_embedding(time_values.unsqueeze(-1))
        
        # Apply temporal convolution
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        
        # Apply temporal attention
        attended_x, _ = self.temporal_attention(x + time_emb, x + time_emb, x)
        return attended_x

class ImprovedHeteroGNN(torch.nn.Module):
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 hidden_channels: int, out_channels: int, feature_dims: Dict[str, int]):
        super().__init__()
        
        # Store metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        
        # Feature encoding
        self.feature_encoders = torch.nn.ModuleDict({
            node_type: ImprovedFeatureEncoder(feature_dims[node_type], hidden_channels)
            for node_type in self.node_types
        })
        
        # Temporal encoding
        self.temporal_encoders = torch.nn.ModuleDict({
            node_type: ImprovedTemporalEncoder(hidden_channels)
            for node_type in self.node_types
        })
        
        # Graph convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First convolution layer with attention
        conv1_dict = {}
        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            conv1_dict[edge_type] = SAGEConv(
                (feature_dims[src_type], feature_dims[dst_type]),
                hidden_channels
            )
        self.convs.append(HeteroConv(conv1_dict, aggr='mean'))
        
        # Second convolution layer with attention
        conv2_dict = {}
        for edge_type in self.edge_types:
            conv2_dict[edge_type] = SAGEConv(
                (hidden_channels, hidden_channels),
                hidden_channels
            )
        self.convs.append(HeteroConv(conv2_dict, aggr='mean'))
        
        # Layer normalization and dropout
        self.layer_norms = torch.nn.ModuleList([
            LayerNorm(hidden_channels) for _ in range(2)
        ])
        self.dropout = Dropout(p=0.2)
        
        # Output layer for study nodes
        self.output = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                time_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Feature encoding
        for node_type in x_dict.keys():
            x_dict[node_type] = self.feature_encoders[node_type](
                x_dict[node_type],
                self.get_feature_types(node_type)
            )
            
            # Temporal encoding if time values exist
            if node_type in time_dict:
                x_dict[node_type] = self.temporal_encoders[node_type](
                    x_dict[node_type],
                    time_dict[node_type]
                )
        
        # Graph convolutions with residual connections
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            for node_type in x_dict_new.keys():
                x_dict_new[node_type] = self.layer_norms[i](x_dict_new[node_type])
                x_dict_new[node_type] = F.relu(x_dict_new[node_type])
                x_dict_new[node_type] = self.dropout(x_dict_new[node_type])
                if node_type in x_dict:  # Add residual connection
                    x_dict_new[node_type] += x_dict[node_type]
            x_dict = x_dict_new
        
        # Return predictions for study nodes
        return self.output(x_dict['study'])
    
    def get_feature_types(self, node_type: str) -> Dict[str, str]:
        # Define feature types for each node type
        feature_types = {
            'study': {
                'enrollment': 'numerical',
                'start_date': 'temporal',
                'study_type': 'categorical'
            },
            'outcome': {
                'date': 'temporal',
                'description': 'categorical'
            },
            'intervention': {
                'date': 'temporal',
                'type': 'categorical'
            },
            'facility': {
                'name': 'categorical',
                'city': 'categorical',
                'country': 'categorical'
            }
        }
        return feature_types.get(node_type, {})

def process_study_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process study features with improved handling of different data types."""
    # Numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_features = df[numerical_cols].fillna(0).values
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_features = []
    for col in categorical_cols:
        if col != 'start_date':  # Handle dates separately
            # Convert categories to indices
            categories = pd.Categorical(df[col].fillna('UNKNOWN'))
            categorical_features.append(categories.codes)
    
    # Temporal features
    if 'start_date' in df.columns:
        dates = pd.to_datetime(df['start_date'], errors='coerce')
        # Convert to days since minimum date
        min_date = dates.min()
        temporal_features = (dates - min_date).dt.days.fillna(0).values
    else:
        temporal_features = np.zeros(len(df))
    
    # Combine features for x
    all_features = np.concatenate([
        numerical_features,
        np.stack(categorical_features, axis=1) if categorical_features else np.zeros((len(df), 0))
    ], axis=1)
    
    return torch.tensor(all_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def process_outcome_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process outcome features."""
    features = []
    
    # Process numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        numerical_features = df[numerical_cols].fillna(0).values
        features.append(numerical_features)
    
    # Process categorical columns (excluding date)
    categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'date']
    for col in categorical_cols:
        categories = pd.Categorical(df[col].fillna('UNKNOWN'))
        features.append(categories.codes.reshape(-1, 1))
    
    # Process date column
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'], errors='coerce')
        min_date = dates.min()
        temporal_features = (dates - min_date).dt.days.fillna(0).values
    else:
        temporal_features = np.zeros(len(df))
    
    # Combine features for x
    combined_features = np.concatenate(features, axis=1) if features else np.zeros((len(df), 1))
    return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def process_intervention_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process intervention features."""
    features = []
    
    # Process numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        numerical_features = df[numerical_cols].fillna(0).values
        features.append(numerical_features)
    
    # Process categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'date':  # Handle dates separately
            categories = pd.Categorical(df[col].fillna('UNKNOWN'))
            features.append(categories.codes.reshape(-1, 1))
    
    # Process date column
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'], errors='coerce')
        min_date = dates.min()
        temporal_features = (dates - min_date).dt.days.fillna(0).values
    else:
        temporal_features = np.zeros(len(df))
    
    # Combine features for x
    combined_features = np.concatenate(features, axis=1) if features else np.zeros((len(df), 1))
    return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def process_facility_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process facility features."""
    features = []
    
    # Process numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        numerical_features = df[numerical_cols].fillna(0).values
        features.append(numerical_features)
    
    # Process categorical columns
    categorical_cols = ['name', 'city', 'country']
    for col in categorical_cols:
        if col in df.columns:
            categories = pd.Categorical(df[col].fillna('UNKNOWN'))
            features.append(categories.codes.reshape(-1, 1))
    
    # No temporal features for facilities
    temporal_features = np.zeros(len(df))
    
    # Combine features for x
    combined_features = np.concatenate(features, axis=1) if features else np.zeros((len(df), 1))
    return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def create_heterogeneous_graph(studies_df: pd.DataFrame, outcomes_df: pd.DataFrame,
                             interventions_df: pd.DataFrame, facilities_df: pd.DataFrame,
                             rdf_graph: Graph) -> HeteroData:
    """Create heterogeneous graph from dataframes and RDF graph."""
    data = HeteroData()
    
    # Create node features with improved processing
    node_features = {}
    time_dict = {}
    
    # Process each node type and store features and temporal values
    for node_type, df, process_fn in [
        ('study', studies_df, process_study_features),
        ('outcome', outcomes_df, process_outcome_features),
        ('intervention', interventions_df, process_intervention_features),
        ('facility', facilities_df, process_facility_features)
    ]:
        features, temporal = process_fn(df)
        node_features[node_type] = features
        time_dict[node_type] = temporal
    
    # Add node features to graph
    for node_type, features in node_features.items():
        data[node_type].x = features
    
    # Add temporal features to graph
    data.time_dict = time_dict
    
    # Add edges from RDF graph
    add_edges_from_rdf(data, rdf_graph, node_features)
    
    return data

class ImprovedTrainer:
    def __init__(self, model: ImprovedHeteroGNN, device: torch.device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=100,
            steps_per_epoch=100,
            pct_start=0.3
        )
        self.early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        self.metrics = {
            'auc': roc_auc_score,
            'ap': average_precision_score,
            'f1': f1_score
        }
        
    def train_epoch(self, data: HeteroData) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with time_dict
        out = self.model(data.x_dict, data.edge_index_dict, data.time_dict)
        loss = F.binary_cross_entropy_with_logits(out, data.y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {'loss': loss.item()}
    
    def validate(self, data: HeteroData) -> Dict[str, float]:
        self.model.eval()
        metrics = {}
        
        with torch.no_grad():
            # Forward pass with time_dict
            out = self.model(data.x_dict, data.edge_index_dict, data.time_dict)
            pred = torch.sigmoid(out).cpu().numpy()
            y_true = data.y.cpu().numpy()
            
            for name, metric_fn in self.metrics.items():
                metrics[name] = metric_fn(y_true, pred)
                
        return metrics

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def load_and_process_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess CSV data."""
    # Load CSV files
    studies_df = pd.read_csv(os.path.join(data_dir, 'studies.csv'))
    outcomes_df = pd.read_csv(os.path.join(data_dir, 'outcomes.csv'))
    interventions_df = pd.read_csv(os.path.join(data_dir, 'interventions.csv'))
    facilities_df = pd.read_csv(os.path.join(data_dir, 'facilities.csv'))
    
    # Preprocess dates
    for df in [studies_df, outcomes_df, interventions_df]:
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return studies_df, outcomes_df, interventions_df, facilities_df

def main():
    # Initialize metrics logging
    logger = MetricsLogger(use_wandb=WANDB_AVAILABLE)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and process data
    studies_df, outcomes_df, interventions_df, facilities_df = load_and_process_data('data')
    rdf_graph = load_rdf_mappings('output')
    
    # Create heterogeneous graph
    data = create_heterogeneous_graph(
        studies_df, outcomes_df, interventions_df, facilities_df, rdf_graph
    )
    
    # Split data
    train_data, val_data, test_data = split_data(data)
    
    # Initialize model
    model = ImprovedHeteroGNN(
        metadata=(data.node_types, data.edge_types),
        hidden_channels=64,
        out_channels=1,
        feature_dims=get_feature_dims(data)
    ).to(device)
    
    # Initialize trainer
    trainer = ImprovedTrainer(model, device)
    
    # Training loop
    best_val_auc = 0
    for epoch in tqdm(range(100), desc="Training"):
        # Train
        train_metrics = trainer.train_epoch(train_data)
        
        # Validate
        val_metrics = trainer.validate(val_data)
        
        # Log metrics
        logger.log({
            'train_loss': train_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_ap': val_metrics['ap'],
            'val_f1': val_metrics['f1']
        })
        
        # Early stopping
        if trainer.early_stopping(val_metrics['auc']):
            print("Early stopping triggered")
            break
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Test final model
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = trainer.validate(test_data)
    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    plot_training_results(model, test_data)
    
    logger.finish()

if __name__ == "__main__":
    main() 