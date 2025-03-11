import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rdflib import Graph, Namespace, URIRef, RDF, Literal
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim import Adam
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_training_data():
    """Create a simple training dataset with timestamps and outcomes."""
    # Create sample training data
    data = {
        'timestamp': pd.date_range(start='2010-01-01', periods=100, freq='M'),
        'nct_id': range(1, 101),
        'outcome': np.random.randint(0, 2, size=100)
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
    """Create a mock RDF graph for testing."""
    g = Graph()
    ns = Namespace("http://example.org/ns#")
    
    # Add some mock triples for each study
    for study_id in task_study_ids:
        study_uri = URIRef(f"http://example.org/study/{study_id}")
        g.add((study_uri, RDF.type, ns.Study))
        g.add((study_uri, ns.nct_id, Literal(study_id)))
        g.add((study_uri, ns.enrollment, Literal(np.random.randint(10, 1000))))
        g.add((study_uri, ns.phase, Literal(f"Phase {np.random.randint(1, 4)}")))
        
        # Add sponsor relationship
        sponsor_id = np.random.randint(1, 20)
        g.add((study_uri, ns.sponsor_id, Literal(sponsor_id)))
        
        # Add condition relationship
        condition_id = np.random.randint(1, 30)
        g.add((study_uri, ns.condition_id, Literal(condition_id)))
        
        # Add intervention relationship
        intervention_id = np.random.randint(1, 25)
        g.add((study_uri, ns.intervention_id, Literal(intervention_id)))
        
        # Add facility relationship
        facility_id = np.random.randint(1, 40)
        g.add((study_uri, ns.facility_id, Literal(facility_id)))
    
    return g

def extract_features_from_graph(g, task_study_ids):
    """Extract features from RDF graph for each study."""
    ns = Namespace("http://example.org/ns#")
    features = {}
    
    for study_id in task_study_ids:
        study_uri = URIRef(f"http://example.org/study/{study_id}")
        
        # Extract numeric features
        enrollment = next(g.objects(study_uri, ns.enrollment), 0)
        sponsor_id = next(g.objects(study_uri, ns.sponsor_id), 0)
        condition_id = next(g.objects(study_uri, ns.condition_id), 0)
        intervention_id = next(g.objects(study_uri, ns.intervention_id), 0)
        facility_id = next(g.objects(study_uri, ns.facility_id), 0)
        
        # Convert phase to numeric
        phase_str = str(next(g.objects(study_uri, ns.phase), "Phase 0"))
        phase = int(phase_str.split()[-1])
        
        # Combine features
        feature_vector = [
            int(enrollment),
            int(sponsor_id),
            int(condition_id),
            int(intervention_id),
            int(facility_id),
            phase
        ]
        features[study_id] = feature_vector
    
    return features

def create_pyg_data(df, features_dict):
    """Create PyTorch Geometric data object."""
    # Get features for studies in this subset
    subset_features = np.array([features_dict[study_id] for study_id in df['nct_id']])
    
    # Create edge index (fully connected graph)
    num_nodes = len(df)
    edge_index = torch.tensor(np.array(np.meshgrid(range(num_nodes), range(num_nodes))), dtype=torch.long)
    
    # Create node features
    x = torch.tensor(subset_features, dtype=torch.float)
    
    # Create target
    y = torch.tensor(df['outcome'].values, dtype=torch.float)
    
    # Create timestamps
    timestamps = torch.tensor(df['timestamp'].values, dtype=torch.float).reshape(-1, 1)
    
    # Combine node features with timestamps
    x = torch.cat([x, timestamps], dim=1)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data = data.to(device)
    
    return data

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Final prediction
        x = self.lin(x)
        return torch.sigmoid(x)

def train_model(model, train_data, val_data, test_data, epochs=100):
    """Train the GNN model."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out, train_data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = criterion(val_out, val_data.y.view(-1, 1))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {loss.item():.4f}")
            print(f"Val Loss: {val_loss.item():.4f}")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(test_data)
        test_loss = criterion(test_out, test_data.y.view(-1, 1))
        test_preds = (test_out > 0.5).float()
        accuracy = (test_preds == test_data.y.view(-1, 1)).float().mean()
        
        print("\nFinal Test Results:")
        print(f"Test Loss: {test_loss.item():.4f}")
        print(f"Test Accuracy: {accuracy.item():.4f}")

def main():
    # Load training data first
    train_df, val_df, test_df = load_training_data()
    
    # Get all unique NCT IDs for feature extraction
    task_study_ids = pd.concat([train_df['nct_id'], val_df['nct_id'], test_df['nct_id']]).unique()
    print(f"\nTotal unique study IDs: {len(task_study_ids)}")
    
    # Create mock RDF graph
    g = create_mock_rdf_graph(task_study_ids)
    print(f"\nCreated mock RDF graph with {len(g)} triples")
    
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
    
    # Initialize model
    input_dim = next(iter(features_dict.values())).__len__() + 1  # features + timestamp
    model = GNN(input_dim=input_dim, hidden_dim=64, output_dim=1)
    print(f"\nModel architecture:\n{model}")
    
    # Train model
    train_model(model, train_data, val_data, test_data)

if __name__ == "__main__":
    main() 