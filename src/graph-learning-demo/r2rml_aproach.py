#!/usr/bin/env python
# coding: utf-8

# # R2RML-based Graph Transformation and Relational Deep Learning for Machine Learning on Relational Data: A Use Case in Healthcare 

# ## Table of Contents
# 
# 1. [Use Case Implementation](#use-case-implementation)
# 2. [Setup and Dependencies](#setup-and-dependencies)
# 3. [Data Loading and Preparation](#data-loading-and-preparation)
# 4. [Graph Construction](#graph-construction)
# 5. [Model Architecture](#model-architecture)
# 6. [Training and Evaluation](#training-and-evaluation)
# 7. [Results and Analysis](#results-and-analysis)

# ### Use Case Implementation: 
# 
# **Objective:**  
# The goal of this use case is to compare two approaches for applying machine learning on relational databases:  
# 1. **Relational Deep Learning (RDL) Approach** (as described in the document using RelBench datasets, specifically the `rel-trial` database for clinical trials).  
# 2. **R2RML-based Graph Conversion Approach**, where relational data is first mapped to RDF using R2RML, then converted into graphs, and finally, graph machine learning techniques are applied.
# 
# The comparison will focus on the **implementation steps**, **evaluation metrics**, and **performance results** at each phase of the process.
# 
# ---
# 
# ### **Implementation Steps:**
# 
# #### **1. Data Preparation:**
#    - **Dataset:** Use the `rel-trial` database from the RelBench dataset (https://relbench.stanford.edu/start/), which contains clinical trial data.
#    - **Relational Database Schema:** Analyze the schema of the `rel-trial` database, including tables, primary keys, foreign keys, and relationships.
#    - **Task Definition:** Define a predictive task (e.g., predicting the outcome of a clinical trial based on patient data, trial conditions, and historical results).
#    - 
# #### **2. Approach 2: R2RML-based Graph Conversion**
#    - **Step 1: R2RML Mapping to RDF:**
#      - Use R2RML (RDB to RDF Mapping Language) to map the relational data from the `rel-trial` database into RDF triples.
#      - **Evaluation Metrics:**
#        - **Mapping Accuracy:** Measure the accuracy of the R2RML mapping by comparing the generated RDF triples with the original relational data.
#        - **Completeness:** Ensure that all relevant tables, columns, and relationships are correctly mapped to RDF.
#        - **Performance:** Measure the time taken to perform the R2RML mapping.
#    - **Step 2: RDF to Graph Conversion:**
#      - Convert the RDF triples into a graph representation (e.g., using tools like RDFLib or a Triple Storage Tool).
#      - **Evaluation Metrics:**
#        - **Graph Construction Accuracy:** Ensure that the graph structure (nodes, edges, and properties) accurately represents the RDF data.
#        - **Graph Size:** Measure the number of nodes and edges in the resulting graph.
#    - **Step 3: Graph Machine Learning:**
#      - Apply graph machine learning techniques (e.g., GNNs) on the constructed graph.
#      - **Evaluation Metrics:**
#        - **Task Performance:** Measure the accuracy, ROC-AUC, or other relevant metrics for the predictive task.
#        - **Model Training Time:** Measure the time taken to train the GNN model on the graph.
# 
# <!-- #### **4. Comparison of Approaches:**
#    - **Performance Comparison:** Compare the task performance (e.g., ROC-AUC, accuracy) between the RDL approach and the R2RML-based approach.
#    - **Efficiency Comparison:** Compare the time taken for data preparation, model training.
#    - **Scalability:** Evaluate how each approach scales with larger datasets (e.g., more tables, more rows).
#    - **Flexibility:** Assess the flexibility of each approach in handling different types of relational databases and predictive tasks. -->
# 
# In summary, the workflow consists of the following steps:
# 1. **Achieving Semantic Data Interoperability**: Transforming JSON input data into RDF, making it machine-readable and semantically enriched.
# 2. **Graph Learning and Visualization**: Constructing and analyzing the RDF graph, with metrics calculation for insights.
# 3. **Metrics Calculation**: Evaluating the performance and utility of the generated RDF graph through visualizations and metrics.


# # Approach 2: R2RML, Graph Mapping and Graph Machine Learning

# - **R2RML Mapping:**
# 
#     - Map RDF triples based on R2RML mappings. R2RML mappings to return RDF data in Turtle format. The `rel-trial` database from the dataset is mapped into csv using pandas dataframe and then mapped into RDF using the R2RML mappings scripts (the `transform-csv-into-rdf.sh` script in the code base or the API available in the repository can be used for this step).
# 
# - **RDF to Graph Conversion:**
# 
#     - The RDF triples are parsed using rdflib and converted into a PyG HeteroData graph. Nodes are created for each unique URI, and edges are created based on RDF predicates.
# 
# - **Graph Machine Learning:**
# 
#     - The GNN model (HeteroGraphSAGE) is applied to the graph. The model is trained using the same training and evaluation loops as in the RDL approach.
# 
# - **Integration with RDL:**
# 
#     - The R2RML-based approach can be compared with the RDL approach by evaluating the performance metrics (e.g., ROC-AUC) on the test set.
#  
# ### Overview
# 
# The implementation:
# - Loads CSV data (studies, outcomes, interventions, facilities)
# - Integrates RDF mappings from the output folder
# - Creates a heterogeneous graph structure
# - Implements a HeteroGNN model with GraphSAGE convolutions
# - Trains the model with proper train/val/test splits
# - Evaluates performance using AUC and AP metrics

# ### Setup and Imports

# In[27]:


import os
import torch
import numpy as np
import pandas as pd
from rdflib import Graph, URIRef, Literal, XSD
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU, Dropout, MultiheadAttention, LSTM
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ### Data Loading Functions
# 
# #### 1. Loading CSV Data
# 
# Loads the clinical trials data from CSV files.

# In[30]:


def load_csv_data():
    """Load CSV data files and RelBench task data."""
    print("\nLoading CSV data...")
    
    # Load CSV files with low_memory=False to handle mixed types
    studies_df = pd.read_csv('data/studies.csv', low_memory=False)
    outcomes_df = pd.read_csv('data/outcomes.csv', low_memory=False)
    interventions_df = pd.read_csv('data/interventions.csv', low_memory=False)
    facilities_df = pd.read_csv('data/facilities.csv', low_memory=False)
    
    print(f"Loaded {len(studies_df)} studies")
    print(f"Loaded {len(outcomes_df)} outcomes")
    print(f"Loaded {len(interventions_df)} interventions")
    print(f"Loaded {len(facilities_df)} facilities")
    
    # Load RelBench dataset and task
    print("\nLoading RelBench dataset and task...")
    dataset = get_dataset("rel-trial", download=True)
    task = get_task("rel-trial", "study-outcome", download=True)
    
    # Get only training data
    train_table = task.get_table("train")
    train_df = train_table.df
    
    print(f"\nLoaded RelBench training data:")
    print(f"Train set size: {len(train_df)}")
    
    # Set validation and test DataFrames to None since we're only using training data
    val_df = None
    test_df = None
    
    return studies_df, outcomes_df, interventions_df, facilities_df, train_df, val_df, test_df

# Load the data
studies_df, outcomes_df, interventions_df, facilities_df, train_df, val_df, test_df = load_csv_data()


# #### 2. Loading RDF Mappings
# 
# Loads and processes RDF mappings from the output folder.

# In[32]:


def load_rdf_mappings(output_folder):
    """Load RDF mappings from the output folder."""
    print("\nLoading RDF mappings...")
    rdf_graph = Graph()
    
    # List of RDF files to load
    rdf_files = [
        'studies-rdf.ttl',
        'interventions-rdf.ttl',
        'facilities-rdf.ttl',
        'outcomes-rdf.ttl',
        'reported_event_totals-rdf.ttl',
        'drop_withdrawals-rdf.ttl',
        'sponsors_studies-rdf.ttl',
        'conditions_studies-rdf.ttl'
    ]
    
    def fix_date(date_str):
        """Fix incomplete date strings."""
        if date_str.endswith('-'):
            return date_str + '01'
        parts = date_str.split('-')
        if len(parts) == 2:
            return date_str + '-01'
        return date_str
    
    for filename in rdf_files:
        filepath = os.path.join(output_folder, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            try:
                # Read file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix date formats
                lines = content.split('\n')
                fixed_lines = []
                for line in lines:
                    if '^^xsd:date' in line:
                        parts = line.split('^^xsd:date')
                        if len(parts) == 2:
                            date_str = parts[0].strip().strip('"')
                            fixed_date = fix_date(date_str)
                            line = f'"{fixed_date}"^^xsd:date{parts[1]}'
                    fixed_lines.append(line)
                
                # Parse fixed content
                rdf_graph.parse(data='\n'.join(fixed_lines), format="turtle")
                print(f"Loaded {len(rdf_graph)} total triples")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        else:
            print(f"Warning: {filename} not found")
    
    return rdf_graph

# Load RDF mappings
rdf_graph = load_rdf_mappings('output')


# ### Graph Data Creation
# 
# Creates a heterogeneous graph structure from the CSV and RDF data.

# In[34]:


def extract_node_id(uri_str):
    """Extract node ID from URI string."""
    try:
        # Try to extract numeric ID from the end of the URI
        parts = str(uri_str).split('/')
        last_part = parts[-1].split('#')[-1]
        if last_part.isdigit():
            return int(last_part)
        # If not numeric, hash the URI to get a consistent ID
        return hash(uri_str) % (10**9)  # Use modulo to keep IDs manageable
    except Exception as e:
        print(f"Error extracting node ID from {uri_str}: {str(e)}")
        return hash(uri_str) % (10**9)

def process_study_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process study features with improved handling of different data types."""
    features = []
    
    # Process numerical features with standardization
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        numerical_features = df[numerical_cols].fillna(df[numerical_cols].mean())
        # Standardize numerical features
        numerical_features = (numerical_features - numerical_features.mean()) / (numerical_features.std() + 1e-8)
        features.append(numerical_features.values)
    
    # Process categorical features with improved encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'start_date':  # Handle dates separately
            # Convert categories to indices with unknown handling
            categories = pd.Categorical(df[col].fillna('UNKNOWN'))
            # Add 1 to avoid 0 index (reserved for padding)
            categorical_features = categories.codes + 1
            features.append(categorical_features.reshape(-1, 1))
    
    # Process temporal features with improved normalization
    if 'start_date' in df.columns:
        dates = pd.to_datetime(df['start_date'], errors='coerce')
        # Convert to days since minimum date and normalize
        min_date = dates.min()
        temporal_features = (dates - min_date).dt.days.fillna(0).values
        temporal_features = (temporal_features - temporal_features.mean()) / (temporal_features.std() + 1e-8)
    else:
        temporal_features = np.zeros(len(df))
    
    # Combine features
    all_features = np.concatenate(features, axis=1) if features else np.zeros((len(df), 1))
    
    return torch.tensor(all_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def process_outcome_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process outcome features with improved encoding."""
    features = []
    
    # Process numerical features with standardization
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        numerical_features = df[numerical_cols].fillna(df[numerical_cols].mean())
        numerical_features = (numerical_features - numerical_features.mean()) / (numerical_features.std() + 1e-8)
        features.append(numerical_features.values)
    
    # Process categorical features with improved encoding
    categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'date']
    for col in categorical_cols:
        categories = pd.Categorical(df[col].fillna('UNKNOWN'))
        categorical_features = categories.codes + 1  # Add 1 to avoid 0 index
        features.append(categorical_features.reshape(-1, 1))
    
    # Process temporal features with improved normalization
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'], errors='coerce')
        min_date = dates.min()
        temporal_features = (dates - min_date).dt.days.fillna(0).values
        temporal_features = (temporal_features - temporal_features.mean()) / (temporal_features.std() + 1e-8)
    else:
        temporal_features = np.zeros(len(df))
    
    # Combine features
    combined_features = np.concatenate(features, axis=1) if features else np.zeros((len(df), 1))
    return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def process_intervention_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process intervention features with improved encoding."""
    features = []
    
    # Process numerical features with standardization
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        numerical_features = df[numerical_cols].fillna(df[numerical_cols].mean())
        numerical_features = (numerical_features - numerical_features.mean()) / (numerical_features.std() + 1e-8)
        features.append(numerical_features.values)
    
    # Process categorical features with improved encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'date':  # Handle dates separately
            categories = pd.Categorical(df[col].fillna('UNKNOWN'))
            categorical_features = categories.codes + 1  # Add 1 to avoid 0 index
            features.append(categorical_features.reshape(-1, 1))
    
    # Process temporal features with improved normalization
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'], errors='coerce')
        min_date = dates.min()
        temporal_features = (dates - min_date).dt.days.fillna(0).values
        temporal_features = (temporal_features - temporal_features.mean()) / (temporal_features.std() + 1e-8)
    else:
        temporal_features = np.zeros(len(df))
    
    # Combine features
    combined_features = np.concatenate(features, axis=1) if features else np.zeros((len(df), 1))
    return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def process_facility_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process facility features with improved encoding."""
    features = []
    
    # Process numerical features with standardization
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        numerical_features = df[numerical_cols].fillna(df[numerical_cols].mean())
        numerical_features = (numerical_features - numerical_features.mean()) / (numerical_features.std() + 1e-8)
        features.append(numerical_features.values)
    
    # Process categorical features with improved encoding
    categorical_cols = ['name', 'city', 'country']
    for col in categorical_cols:
        if col in df.columns:
            categories = pd.Categorical(df[col].fillna('UNKNOWN'))
            categorical_features = categories.codes + 1  # Add 1 to avoid 0 index
            features.append(categorical_features.reshape(-1, 1))
    
    # No temporal features for facilities
    temporal_features = np.zeros(len(df))
    
    # Combine features
    combined_features = np.concatenate(features, axis=1) if features else np.zeros((len(df), 1))
    return torch.tensor(combined_features, dtype=torch.float32), torch.tensor(temporal_features, dtype=torch.float32)

def create_graph_data(studies_df, outcomes_df, interventions_df, facilities_df, rdf_graph, train_df, val_df, test_df):
    """Create heterogeneous graph data from CSV and RDF data."""
    print("\nCreating heterogeneous graph data...")
    data = HeteroData()
    
    # Print column names for debugging
    print("\nStudies DataFrame columns:")
    print(studies_df.columns.tolist())
    print("\nTrain DataFrame columns:")
    print(train_df.columns.tolist())
    
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
    
    # Create labels from RelBench data
    num_studies = len(studies_df)
    study_labels = torch.zeros((num_studies, 1), dtype=torch.float32)
    
    # Map study IDs to indices using nct_id
    study_id_to_idx = {nct_id: idx for idx, nct_id in enumerate(studies_df['nct_id'])}
    
    # Set labels for train set
    for _, row in train_df.iterrows():
        nct_id = row['nct_id']
        if nct_id in study_id_to_idx:
            study_labels[study_id_to_idx[nct_id]] = row['outcome']  # Use 'outcome' column as label
    
    data['study'].y = study_labels
    
    print("\nNode feature dimensions:")
    for node_type, features in node_features.items():
        print(f"{node_type} features: {features.shape}")
        print(f"{node_type} temporal: {time_dict[node_type].shape}")
    
    # Extract edges from RDF graph
    edges_by_type = {}
    print("\nExtracting edges...")
    
    # Helper function to get node type
    def get_node_type(uri):
        uri_str = str(uri)
        for node_type in node_features.keys():
            if node_type in uri_str:
                return node_type
        return None
    
    # Process edges
    for s, p, o in rdf_graph:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            s_type = get_node_type(s)
            o_type = get_node_type(o)
            
            if s_type is None or o_type is None:
                continue
            
            # Extract edge type from predicate
            edge_type = str(p).split('/')[-1].split('#')[-1]
            if edge_type == 'type':
                continue  # Skip rdf:type edges
            
            edge_key = (s_type, edge_type, o_type)
            if edge_key not in edges_by_type:
                edges_by_type[edge_key] = []
            
            # Try to get node indices
            try:
                s_idx = int(str(s).split('/')[-1])
                o_idx = int(str(o).split('/')[-1])
                
                # Verify indices are within bounds
                if (s_idx < len(node_features[s_type]) and 
                    o_idx < len(node_features[o_type])):
                    edges_by_type[edge_key].append((s_idx, o_idx))
            except (ValueError, IndexError):
                continue
    
    # Add self-loops for all node types
    for node_type in node_features.keys():
        edge_key = (node_type, 'self', node_type)
        self_loops = [(i, i) for i in range(len(node_features[node_type]))]
        edges_by_type[edge_key] = self_loops
    
    # Add edges to HeteroData
    print("\nAdding edges to graph...")
    for (s_type, edge_type, o_type), edges in edges_by_type.items():
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data[s_type, edge_type, o_type].edge_index = edge_index
            print(f"Added {len(edges)} edges of type ({s_type}, {edge_type}, {o_type})")
    
    # Create train mask only
    train_mask = torch.zeros(num_studies, dtype=torch.bool)
    
    # Set mask based on RelBench training data
    for _, row in train_df.iterrows():
        nct_id = row['nct_id']
        if nct_id in study_id_to_idx:
            train_mask[study_id_to_idx[nct_id]] = True
    
    # Add train mask to data
    data['study'].train_mask = train_mask
    
    return data

# Create the graph data
data = create_graph_data(studies_df, outcomes_df, interventions_df, facilities_df, rdf_graph, train_df, val_df, test_df)


# ### Model Architecture
# 
# Implements a HeteroGNN model using GraphSAGE convolutions.

# In[36]:


class ImprovedFeatureEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature type-specific encoders
        self.numerical_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )
        
        # Increased embedding size and added padding_idx for unknown values
        self.categorical_encoder = torch.nn.Embedding(
            num_embeddings=10000,  # Increased from 1000
            embedding_dim=hidden_dim,
            padding_idx=0  # Use 0 for padding/unknown values
        )
        
        self.temporal_encoder = torch.nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Multi-head attention for feature combination
        self.feature_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, feature_types: Dict[str, str]) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_features = []
        
        # Process each feature based on its type
        for i, (feat_name, feat_type) in enumerate(feature_types.items()):
            if feat_type == 'numerical':
                # Reshape numerical features for processing
                feat = x[:, i:i+1]  # Keep as 2D tensor
                encoded = self.numerical_encoder(feat)
            elif feat_type == 'categorical':
                # Convert to long type and clamp indices
                feat = x[:, i].long()
                feat = torch.clamp(feat, min=0, max=9999)  # Ensure indices are within bounds
                encoded = self.categorical_encoder(feat)
            elif feat_type == 'temporal':
                # Reshape temporal features for LSTM
                feat = x[:, i:i+1]  # Keep as 2D tensor
                encoded, _ = self.temporal_encoder(feat.unsqueeze(1))
                encoded = encoded.squeeze(1)
            else:
                continue
                
            encoded_features.append(encoded)
        
        # Stack encoded features
        if encoded_features:
            features_stack = torch.stack(encoded_features, dim=1)
            
            # Apply multi-head attention
            attended_features, _ = self.feature_attention(
                features_stack,
                features_stack,
                features_stack
            )
            
            # Combine attended features
            combined = attended_features.mean(dim=1)
        else:
            # If no features were processed, create zero tensor
            combined = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Final projection
        return self.output_proj(combined)

class ImprovedTemporalEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embedding = torch.nn.Linear(1, hidden_dim)
        
        # Temporal attention
        self.temporal_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Layer normalization and projection
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, time_values: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, hidden_dim]
        # time_values shape: [batch_size]
        
        # Encode time values
        time_emb = self.time_embedding(time_values.unsqueeze(-1))  # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        time_emb = time_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply temporal attention
        attended_x, _ = self.temporal_attention(
            x + time_emb,  # query
            x + time_emb,  # key
            x  # value
        )
        
        # Remove sequence dimension and apply layer norm
        x = attended_x.squeeze(1)  # [batch_size, hidden_dim]
        x = self.layer_norm(x)
        
        # Final projection
        return self.output_proj(x)

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, feature_dims):
        super().__init__()
        
        # Store metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        
        print("\nInitializing HeteroGNN with:")
        print(f"Node types: {self.node_types}")
        print(f"Edge types: {self.edge_types}")
        print(f"Feature dimensions: {feature_dims}")
        
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
        
        # Create convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First convolution layer with attention
        conv1_dict = {}
        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            conv1_dict[edge_type] = SAGEConv(
                (hidden_channels, hidden_channels),  # Use hidden_channels instead of feature_dims
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
    
    def forward(self, x_dict, edge_index_dict, time_dict):
        # Feature encoding
        encoded_dict = {}
        for node_type in x_dict.keys():
            # First encode features
            encoded_dict[node_type] = self.feature_encoders[node_type](
                x_dict[node_type],
                self.get_feature_types(node_type)
            )
            
            # Then apply temporal encoding if time values exist
            if node_type in time_dict:
                encoded_dict[node_type] = self.temporal_encoders[node_type](
                    encoded_dict[node_type],
                    time_dict[node_type]
                )
        
        # Graph convolutions with residual connections
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(encoded_dict, edge_index_dict)
            for node_type in x_dict_new.keys():
                x_dict_new[node_type] = self.layer_norms[i](x_dict_new[node_type])
                x_dict_new[node_type] = F.relu(x_dict_new[node_type])
                x_dict_new[node_type] = self.dropout(x_dict_new[node_type])
                if node_type in encoded_dict:  # Add residual connection
                    x_dict_new[node_type] += encoded_dict[node_type]
            encoded_dict = x_dict_new
        
        # Return predictions for study nodes
        return self.output(encoded_dict['study'])
    
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


# ### Training and Evaluation
# 
# #### 1. Training Function
# 
# Implements the training loop with validation and early stopping.

# In[38]:


class EarlyStopping:
    """Early stopping to stop the training when the loss does not improve after
    certain epochs."""
    def __init__(self, patience=7, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is not improving
        :param min_delta: minimum difference between new loss and old loss for
                         an improvement to be registered
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss):
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

def train_model(model, train_data, val_data, test_data, num_epochs=100, lr=0.01):
    """Train the model using train/val/test splits."""
    print("\nTraining model...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=100,
        pct_start=0.3
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    train_losses = []
    val_metrics = []
    best_val_auc = 0
    best_model = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(train_data.x_dict, train_data.edge_index_dict, train_data.time_dict)
        loss = criterion(out, train_data['study'].y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x_dict, val_data.edge_index_dict, val_data.time_dict)
            val_loss = criterion(val_out, val_data['study'].y)
            
            val_pred = torch.sigmoid(val_out).cpu().numpy()
            val_true = val_data['study'].y.cpu().numpy()
            
            val_auc = roc_auc_score(val_true, val_pred)
            val_ap = average_precision_score(val_true, val_pred)
            
            val_metrics.append({
                'loss': val_loss.item(),
                'auc': val_auc,
                'ap': val_ap
            })
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}:')
            print(f'Train Loss: {loss.item():.4f}')
            print(f'Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
        
        # Early stopping
        if early_stopping(val_auc):
            print("Early stopping triggered")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x_dict, test_data.edge_index_dict, test_data.time_dict)
        test_loss = criterion(test_out, test_data['study'].y)
        
        test_pred = torch.sigmoid(test_out).cpu().numpy()
        test_true = test_data['study'].y.cpu().numpy()
        
        test_auc = roc_auc_score(test_true, test_pred)
        test_ap = average_precision_score(test_true, test_pred)
    
    print('\nTest Results:')
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}')
    
    return train_losses, val_metrics, test_auc, test_ap


# #### 2. Visualization Function
# 
# Plots training and validation metrics.

# In[44]:


import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Function to compute classification metrics
def compute_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute classification metrics for binary/multi-class classification tasks.
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1-Score
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted")

    # ROC-AUC (only for binary classification)
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics["roc_auc"] = auc(fpr, tpr)

    # Confusion Matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """
    Plot the ROC curve for binary classification.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """
    Plot the Precision-Recall curve for binary classification.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()

# Function to plot metrics
def plot_metrics(train_losses, val_metrics, test_data, model):
    """Plot training metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot([m['loss'] for m in val_metrics], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    # Plot AUC
    axes[1].plot([m['auc'] for m in val_metrics])
    axes[1].set_title('Validation AUC')

    # Plot AP
    axes[2].plot([m['ap'] for m in val_metrics])
    axes[2].set_title('Validation AP')

    plt.tight_layout()
    plt.show()

    # Plot ROC curve, Confusion Matrix, etc.
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x_dict, test_data.edge_index_dict, test_data.time_dict)
        test_pred = torch.sigmoid(test_out).cpu().numpy()
        test_true = test_data['study'].y.cpu().numpy()

        # Compute classification metrics
        classification_metrics = compute_classification_metrics(test_true, (test_pred > 0.5).astype(int), test_pred)
        print("\nClassification Metrics:")
        for metric, value in classification_metrics.items():
            if metric != "confusion_matrix":
                print(f"{metric}: {value:.4f}")

        # Plot confusion matrix
        plot_confusion_matrix(test_true, (test_pred > 0.5).astype(int), title="Confusion Matrix (Test Set)")

        # Plot ROC curve
        plot_roc_curve(test_true, test_pred, title="ROC Curve (Test Set)")

        # Plot Precision-Recall curve
        plot_precision_recall_curve(test_true, test_pred, title="Precision-Recall Curve (Test Set)")


# ### Main Execution
# 
# Run the complete training pipeline.

# In[47]:


def main():
    # Get feature dimensions for each node type
    feature_dims = {
        'study': data['study'].x.shape[1],
        'outcome': data['outcome'].x.shape[1],
        'intervention': data['intervention'].x.shape[1],
        'facility': data['facility'].x.shape[1]
    }
    
    print("\nFeature dimensions:")
    for node_type, dim in feature_dims.items():
        print(f"{node_type}: {dim}")
    
    # Create training data only
    train_data = data.clone()
    
    # Move data to device
    train_data = train_data.to(device)
    
    # Initialize model
    model = HeteroGNN(
        metadata=data.metadata(),
        hidden_channels=64,
        out_channels=1,
        feature_dims=feature_dims
    ).to(device)
    
    # Train model using only training data
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\nTraining model...")
    model.train()
    
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(train_data.x_dict, train_data.edge_index_dict, train_data.time_dict)
        loss = criterion(out[train_data['study'].train_mask], train_data['study'].y[train_data['study'].train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    print("\nTraining completed")
    
    return model
        

# Run the main function
main()


# ## **Relevant Resources:**

# 1. **RelBench Dataset and Framework:**  
#    - Website: https://relbench.stanford.edu/  
#    - Documentation: https://relbench.stanford.edu/start/  
#    - GitHub Repository: https://github.com/snap-stanford/relbench  
# 
# 2. **R2RML (RDB to RDF Mapping Language):**  
#    - W3C Specification: https://www.w3.org/TR/r2rml/  
#    - Tools:  
#      - **RMLMapper:** https://github.com/RMLio/rmlmapper-java  
#      - **Ontop:** https://ontop-vkg.org/  
#      - **Apache Jena:** https://jena.apache.org/  
# 
# 3. **Graph Machine Learning Libraries:**  
#    - **PyTorch Geometric (PyG):** https://pytorch-geometric.readthedocs.io/  
#    - **DGL (Deep Graph Library):** https://www.dgl.ai/  
#    - **Graph Neural Networks (GNNs):** https://distill.pub/2021/gnn-intro/  
# 
# 4. **RDF to Graph Conversion Tools:**  
#    - **RDFLib:** https://rdflib.readthedocs.io/  
#    - **Apache Jena:** https://jena.apache.org/  
# 
# 5. **Evaluation Metrics for Machine Learning:**  
#    - **ROC-AUC:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html  
#    - **Accuracy:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html  
#    - **Precision, Recall, F1-Score:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html  
# 
# ---

# In[ ]:




