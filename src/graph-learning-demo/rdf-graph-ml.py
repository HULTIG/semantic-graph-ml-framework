# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import copy
import optuna
from optuna.trial import TrialState
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout, ModuleList, LayerNorm, Sequential, ReLU, ModuleDict
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch.optim import Adam
from tqdm import tqdm

# RDF and data processing imports
from rdflib import Graph, URIRef, Namespace, RDF, Literal
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, accuracy_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

def load_training_data():
    """Create a simple training dataset with timestamps and outcomes."""
    # Create sample training data with meaningful patterns
    n_samples = 200
    data = {
        'timestamp': pd.date_range(start='2010-01-01', periods=n_samples, freq='M'),
        'nct_id': range(1, n_samples + 1),
    }
    train_data = pd.DataFrame(data)
    
    # Convert date to timestamp
    train_data['timestamp'] = train_data['timestamp'].astype(np.int64) // 10**9
    
    # Split into train and test sets
    train_df, test_df = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Create validation set from train
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    print(f"\nLoaded training data:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Print sample of the data
    print("\nSample of training data:")
    print(train_df.head())
    
    return train_df, val_df, test_df

def create_mock_rdf_graph(task_study_ids):
    """Create a mock RDF graph for testing with meaningful patterns."""
    g = Graph()
    ns = Namespace("http://example.org/ns#")
    
    # Create patterns that will influence the outcome
    # Pattern 1: High enrollment + Phase 3 -> More likely to succeed
    # Pattern 2: More facilities -> More likely to succeed
    # Pattern 3: Certain sponsor types -> More likely to succeed
    outcomes = {}
    
    # Add some mock triples for each study
    for study_id in task_study_ids:
        study_uri = URIRef(f"http://example.org/study/{study_id}")
        g.add((study_uri, RDF.type, ns.Study))
        g.add((study_uri, ns.nct_id, Literal(study_id)))
        
        # Generate correlated features
        enrollment = np.random.randint(10, 1000)
        phase = np.random.randint(1, 4)
        sponsor_id = np.random.randint(1, 20)
        condition_id = np.random.randint(1, 30)
        intervention_id = np.random.randint(1, 25)
        facility_count = np.random.randint(1, 40)
        
        # Add triples
        g.add((study_uri, ns.enrollment, Literal(enrollment)))
        g.add((study_uri, ns.phase, Literal(f"Phase {phase}")))
        g.add((study_uri, ns.sponsor_id, Literal(sponsor_id)))
        g.add((study_uri, ns.condition_id, Literal(condition_id)))
        g.add((study_uri, ns.intervention_id, Literal(intervention_id)))
        g.add((study_uri, ns.facility_id, Literal(facility_count)))
        
        # Generate outcome based on patterns
        success_prob = 0.0
        
        # Pattern 1: High enrollment + Phase 3
        if enrollment > 500 and phase == 3:
            success_prob += 0.4
        
        # Pattern 2: More facilities
        if facility_count > 20:
            success_prob += 0.3
        
        # Pattern 3: Certain sponsor types
        if sponsor_id in [1, 5, 10, 15]:  # "Good" sponsors
            success_prob += 0.3
        
        # Add some randomness
        success_prob = min(max(success_prob + np.random.normal(0, 0.1), 0), 1)
        
        # Generate binary outcome
        outcome = int(success_prob > 0.5)
        outcomes[study_id] = outcome
    
    return g, outcomes

def extract_features_from_graph(g, task_study_ids):
    """Extract features from RDF graph for each study with enhanced feature engineering."""
    ns = Namespace("http://example.org/ns#")
    features = {}
    
    # Calculate global statistics for normalization
    all_enrollments = []
    all_facility_counts = []
    all_phases = []
    
    # First pass to collect statistics
    for study_id in task_study_ids:
        study_uri = URIRef(f"http://example.org/study/{study_id}")
        enrollment = int(next(g.objects(study_uri, ns.enrollment), 0))
        facility_count = int(next(g.objects(study_uri, ns.facility_id), 0))
        phase_str = str(next(g.objects(study_uri, ns.phase), "Phase 0"))
        phase = int(phase_str.split()[-1])
        
        all_enrollments.append(enrollment)
        all_facility_counts.append(facility_count)
        all_phases.append(phase)
    
    # Calculate statistics
    mean_enrollment = np.mean(all_enrollments)
    std_enrollment = np.std(all_enrollments) + 1e-8
    mean_facility_count = np.mean(all_facility_counts)
    std_facility_count = np.std(all_facility_counts) + 1e-8
    
    # Second pass to extract and engineer features
    for study_id in task_study_ids:
        study_uri = URIRef(f"http://example.org/study/{study_id}")
        
        # Base features
        enrollment = int(next(g.objects(study_uri, ns.enrollment), 0))
        phase = int(str(next(g.objects(study_uri, ns.phase), "Phase 0")).split()[-1])
        sponsor_id = int(next(g.objects(study_uri, ns.sponsor_id), 0))
        condition_id = int(next(g.objects(study_uri, ns.condition_id), 0))
        intervention_id = int(next(g.objects(study_uri, ns.intervention_id), 0))
        facility_count = int(next(g.objects(study_uri, ns.facility_id), 0))
        
        # Normalize continuous features
        enrollment_norm = (enrollment - mean_enrollment) / std_enrollment
        facility_count_norm = (facility_count - mean_facility_count) / std_facility_count
        
        # Feature interactions
        enrollment_per_facility = enrollment / (facility_count + 1)  # Add 1 to avoid division by zero
        phase_enrollment_interaction = phase * enrollment_norm
        
        # Categorical encodings
        is_phase_3 = float(phase == 3)
        is_good_sponsor = float(sponsor_id in [1, 5, 10, 15])
        
        # Compound features
        complexity_score = (enrollment_norm + facility_count_norm + phase) / 3
        scale_score = np.log1p(enrollment) * np.log1p(facility_count)
        
        # Risk factors
        high_complexity = float(complexity_score > 1.0)
        large_scale = float(scale_score > np.median([s for s in [scale_score] if s > 0]))
        
        # Combine all features
        feature_vector = [
            # Base features (normalized)
            enrollment_norm,
            facility_count_norm,
            phase,
            sponsor_id,
            condition_id,
            intervention_id,
            
            # Feature interactions
            enrollment_per_facility,
            phase_enrollment_interaction,
            
            # Categorical indicators
            is_phase_3,
            is_good_sponsor,
            
            # Compound features
            complexity_score,
            scale_score,
            
            # Risk factors
            high_complexity,
            large_scale,
            
            # Raw values for reference
            enrollment,
            facility_count
        ]
        
        features[study_id] = feature_vector
    
    return features

def create_pyg_data(df, features_dict):
    """Create PyTorch Geometric data object with enhanced features."""
    # Get features for studies in this subset
    subset_features = np.array([features_dict[study_id] for study_id in df['nct_id']])
    
    # Normalize features (excluding binary indicators)
    scaler = StandardScaler()
    continuous_mask = ~np.isin(range(subset_features.shape[1]), [8, 9, 12, 13])  # Indices of binary features
    subset_features[:, continuous_mask] = scaler.fit_transform(subset_features[:, continuous_mask])
    
    # Create temporal features
    timestamps = df['timestamp'].values
    temporal_features = []
    
    for t in timestamps:
        # Convert timestamp to datetime for easier manipulation
        dt = pd.Timestamp(t, unit='s')
        
        temporal_features.append([
            dt.year - 2010,  # Years since 2010
            np.sin(2 * np.pi * dt.month / 12),  # Month seasonality (sin)
            np.cos(2 * np.pi * dt.month / 12),  # Month seasonality (cos)
            dt.quarter,  # Quarter
            dt.is_quarter_end,  # Quarter end indicator
            dt.dayofweek,  # Day of week
        ])
    
    temporal_features = np.array(temporal_features)
    
    # Normalize temporal features (excluding binary indicators)
    temporal_features[:, [0, 3, 5]] = scaler.fit_transform(temporal_features[:, [0, 3, 5]])
    
    # Create edge index based on feature similarity
    num_nodes = len(df)
    edge_list = []
    edge_weights = []
    
    # Create edges between similar studies
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Calculate similarity based on continuous features
            sim = np.corrcoef(subset_features[i, continuous_mask], subset_features[j, continuous_mask])[0, 1]
            
            # Add edge if similarity is high enough
            if not np.isnan(sim) and abs(sim) > 0.5:
                edge_list.extend([[i, j], [j, i]])  # Add both directions
                edge_weights.extend([abs(sim), abs(sim)])
    
    # If no edges meet the similarity threshold, create a minimum spanning tree
    if not edge_list:
        for i in range(num_nodes - 1):
            edge_list.extend([[i, i + 1], [i + 1, i]])
            edge_weights.extend([1.0, 1.0])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    # Combine all features
    x = torch.tensor(np.hstack([subset_features, temporal_features]), dtype=torch.float)
    y = torch.tensor(df['outcome'].values, dtype=torch.float)
    
    # Create PyG data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weights.view(-1, 1),
        y=y
    )
    data = data.to(device)
    
    return data

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.5, use_batch_norm=True):
        super(GNN, self).__init__()
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        )
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                dropout=dropout_rate,
                concat=False  # Average the attention heads
            )
            self.convs.append(conv)
            if use_batch_norm:
                self.convs.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layers with skip connection
        self.skip_connection = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initial feature encoding
        x_encoded = self.feature_encoder(x)
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            if isinstance(conv, GATConv):
                x_encoded = conv(x_encoded, edge_index)
                x_encoded = F.elu(x_encoded)
            else:  # BatchNorm
                x_encoded = conv(x_encoded)
        
        # Skip connection
        x_combined = torch.cat([x_encoded, x], dim=1)
        x_skip = F.elu(self.skip_connection(x_combined))
        
        # Output
        out = self.output(x_skip)
        return torch.sigmoid(out)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate and return a dictionary of metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['avg_precision'] = average_precision_score(y_true, y_prob)
    
    return metrics

def print_metrics(phase, epoch, loss, metrics):
    """Print metrics in a formatted way."""
    print(f"\n{phase} Metrics - Epoch {epoch}:")
    print("─" * 40)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Avg Precision: {metrics['avg_precision']:.4f}")
    print("─" * 40)

def display_results_table(train_metrics, val_metrics, test_metrics, train_loss, val_loss, test_loss=None):
    """Display results in a nicely formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Metric':20} {'Training':15} {'Validation':15} {'Test':15}")
    print("-" * 80)
    
    # Display loss
    test_loss_str = f"{test_loss:.4f}" if test_loss is not None else "N/A"
    print(f"{'Loss':20} {train_loss:15.4f} {val_loss:15.4f} {test_loss_str:>15}")
    
    # Common metrics
    metrics = ['accuracy', 'f1', 'roc_auc', 'avg_precision']
    metric_names = {
        'accuracy': 'Accuracy',
        'f1': 'F1-Score',
        'roc_auc': 'ROC-AUC',
        'avg_precision': 'Avg Precision'
    }
    
    for metric in metrics:
        if metric in train_metrics:
            train_val = f"{train_metrics[metric]:.4f}"
            val_val = f"{val_metrics[metric]:.4f}"
            test_val = f"{test_metrics[metric]:.4f}" if test_metrics else "N/A"
            print(f"{metric_names[metric]:20} {train_val:>15} {val_val:>15} {test_val:>15}")
    
    print("=" * 80)

def objective(trial, train_data, val_data, test_data, input_dim):
    # Hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Initialize model
    model = GNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(train_data)
        
        # Compute focal loss
        y = train_data.y.float().view(-1, 1)
        alpha = 0.75  # weight for class imbalance
        gamma = 2.0   # focusing parameter
        bce = F.binary_cross_entropy(out, y, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = alpha * (1-pt)**gamma * bce
        loss = focal_loss.mean()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = F.binary_cross_entropy(val_out, val_data.y.float().view(-1, 1))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
    
    return best_val_loss.item()

def optimize_hyperparameters(train_data, val_data, test_data, input_dim, n_trials=50):
    """Run hyperparameter optimization."""
    print("\nOptimizing hyperparameters...")
    print("═" * 80)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, test_data, input_dim),
        n_trials=n_trials
    )
    
    print("\nBest trial:")
    print("═" * 40)
    trial = study.best_trial
    print(f"Value: {trial.value:.4f}")
    print("\nBest hyperparameters:")
    print("─" * 40)
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    print("═" * 40)
    
    return trial.params

def train_model_with_params(model, train_data, val_data, test_data, learning_rate, weight_decay):
    """Train the GNN model with optimized parameters."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Calculate class weights for balanced loss
    pos_weight = torch.tensor(
        [(1 - train_data.y.mean()) / train_data.y.mean()]
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_metrics = None
    best_epoch = 0
    
    print("\nTraining Progress:")
    print("═" * 80)
    
    for epoch in tqdm(range(100), desc="Training"):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out, train_data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        # Calculate training metrics
        train_preds = (torch.sigmoid(out) > 0.5).float().cpu().detach().numpy()
        train_probs = torch.sigmoid(out).cpu().detach().numpy()
        train_metrics = calculate_metrics(
            train_data.y.cpu().numpy(),
            train_preds,
            train_probs
        )
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = criterion(val_out, val_data.y.view(-1, 1))
            
            # Calculate validation metrics
            val_preds = (torch.sigmoid(val_out) > 0.5).float().cpu().numpy()
            val_probs = torch.sigmoid(val_out).cpu().numpy()
            val_metrics = calculate_metrics(
                val_data.y.cpu().numpy(),
                val_preds,
                val_probs
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best performance was at epoch {best_epoch}")
                break
        
        if epoch % 10 == 0:
            print_metrics("Training", epoch, loss.item(), train_metrics)
            print_metrics("Validation", epoch, val_loss.item(), val_metrics)
    
    print("\nTraining completed!")
    print("═" * 80)
    print(f"Best Validation Metrics (Epoch {best_epoch}):")
    print("─" * 40)
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("═" * 80)
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(test_data)
        test_loss = criterion(test_out, test_data.y.view(-1, 1))
        test_preds = (torch.sigmoid(test_out) > 0.5).float().cpu().numpy()
        test_probs = torch.sigmoid(test_out).cpu().numpy()
        
        test_metrics = calculate_metrics(
            test_data.y.cpu().numpy(),
            test_preds,
            test_probs
        )
        
        print("\nFinal Test Results:")
        print("═" * 80)
        print_metrics("Test", "Final", test_loss.item(), test_metrics)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(test_data.y.cpu().numpy(), test_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {test_metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load training data first
    train_df, val_df, test_df = load_training_data()
    
    # Get all unique NCT IDs for feature extraction
    task_study_ids = pd.concat([train_df['nct_id'], val_df['nct_id'], test_df['nct_id']]).unique()
    print(f"\nTotal unique study IDs: {len(task_study_ids)}")
    
    # Create mock RDF graph with outcomes
    g, outcomes = create_mock_rdf_graph(task_study_ids)
    print(f"\nCreated mock RDF graph with {len(g)} triples")
    
    # Add outcomes to dataframes
    train_df['outcome'] = train_df['nct_id'].map(outcomes)
    val_df['outcome'] = val_df['nct_id'].map(outcomes)
    test_df['outcome'] = test_df['nct_id'].map(outcomes)
    
    # Print outcome distribution
    print("\nOutcome distribution:")
    print("Train:", train_df['outcome'].value_counts(normalize=True).to_frame('proportion'))
    print("Val:", val_df['outcome'].value_counts(normalize=True).to_frame('proportion'))
    print("Test:", test_df['outcome'].value_counts(normalize=True).to_frame('proportion'))
    
    # Extract features from RDF graph
    features_dict = extract_features_from_graph(g, task_study_ids)
    print(f"\nExtracted features for {len(features_dict)} studies")
    
    # Create PyTorch Geometric data objects
    train_data = create_pyg_data(train_df, features_dict)
    val_data = create_pyg_data(val_df, features_dict)
    test_data = create_pyg_data(test_df, features_dict)
    
    print("\nData objects created:")
    print(f"Train data: {train_data}")
    print(f"Validation data: {val_data}")
    print(f"Test data: {test_data}")
    
    # Get input dimension from data
    input_dim = train_data.x.size(1)
    print(f"\nFeature dimension: {input_dim}")
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(train_data, val_data, test_data, input_dim)
    print("\nBest hyperparameters:", best_params)
    
    # Train final model with best parameters
    model = GNN(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        use_batch_norm=best_params['use_batch_norm']
    ).to(device)
    
    # Convert data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Train model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    print("\nTraining final model...")
    train_losses = []
    train_metrics_history = []
    val_metrics_history = []
    best_val_loss = float('inf')
    best_val_metrics = None
    best_train_metrics = None
    best_train_loss = None
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(train_data)
        y = train_data.y.float().view(-1, 1)
        
        # Compute focal loss
        alpha = 0.75
        gamma = 2.0
        bce = F.binary_cross_entropy(out, y, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = alpha * (1-pt)**gamma * bce
        loss = focal_loss.mean()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Training metrics
                train_preds = (out > 0.5).float()
                train_metrics = calculate_metrics(
                    y.cpu().numpy(),
                    train_preds.cpu().numpy(),
                    out.cpu().numpy()
                )
                
                # Validation metrics
                val_out = model(val_data)
                val_loss = F.binary_cross_entropy(
                    val_out,
                    val_data.y.float().view(-1, 1)
                )
                val_preds = (val_out > 0.5).float()
                val_metrics = calculate_metrics(
                    val_data.y.cpu().numpy(),
                    val_preds.cpu().numpy(),
                    val_out.cpu().numpy()
                )
                
                # Store best metrics
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_metrics = val_metrics
                    best_train_metrics = train_metrics
                    best_train_loss = loss.item()
                
                # Display current metrics
                display_results_table(
                    train_metrics,
                    val_metrics,
                    None,
                    loss.item(),
                    val_loss.item()
                )
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(test_data)
        test_loss = F.binary_cross_entropy(
            test_out,
            test_data.y.float().view(-1, 1)
        )
        test_preds = (test_out > 0.5).float()
        test_metrics = calculate_metrics(
            test_data.y.cpu().numpy(),
            test_preds.cpu().numpy(),
            test_out.cpu().numpy()
        )
        
        print("\nFinal Results:")
        display_results_table(
            best_train_metrics,
            best_val_metrics,
            test_metrics,
            best_train_loss,
            best_val_loss,
            test_loss.item()
        )

if __name__ == "__main__":
    main() 