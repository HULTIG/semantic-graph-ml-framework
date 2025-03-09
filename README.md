# R2RML-based Graph Transformation and Relational Deep Learning Framework

This repository implements a novel framework for applying machine learning on relational databases using two distinct approaches:
1. **Relational Deep Learning (RDL)** - Direct application of graph neural networks on relational data
2. **R2RML-based Graph Transformation** - Converting relational data to RDF graphs for machine learning

## Project Structure

```
src/
├── RelationalToRDFAPI/     # API for converting relational data to RDF
├── graph-learning-demo/    # Jupyter notebooks demonstrating the approaches
├── data/                   # Data storage directory
├── fuseki-config/         # Apache Fuseki configuration files
├── output/                # Generated outputs and results
└── docker-compose.yaml    # Docker configuration for services
```

## Key Components

### 1. Relational to RDF Transformation (RelationalToRDFAPI)

A Java-based API that handles the conversion of relational data to RDF format using R2RML mappings. Features include:
- R2RML mapping execution
- RDF triple generation
- Apache Fuseki integration for RDF storage
- RESTful API endpoints for data transformation

### 2. Graph Learning Implementation

The framework implements two main approaches for machine learning on relational data:

#### a) Relational Deep Learning (RDL)
- Direct application of GNNs on relational databases
- Heterogeneous graph construction from database schema
- Feature engineering for different data types
- Implementation of HeteroGraphSAGE architecture

#### b) R2RML-based Graph Learning
- Two-step process: R2RML mapping followed by graph learning
- RDF graph construction from relational data
- Graph neural network application on RDF structures

## Getting Started

### Prerequisites
- Python 3.8+
- Java 11+
- Docker and Docker Compose
- Apache Fuseki Server

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Set up environment variables:
```bash
cp src/.env.example src/.env
cp src/RelationalToRDFAPI/.env.example src/RelationalToRDFAPI/.env
```

3. Start the services:
```bash
cd src
docker-compose up -d
```

### Running the Demo

1. Navigate to the demo directory:
```bash
cd src/graph-learning-demo
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `demo.ipynb` to explore the implementation

## Implementation Details

### Model Architecture

The framework implements a heterogeneous graph neural network with:
- HeteroEncoder for feature processing
- HeteroGraphSAGE for message passing
- HeteroTemporalEncoder for temporal data
- MLP layers for final predictions

### Performance Metrics

The framework evaluates performance using:
- ROC-AUC score
- Precision-Recall metrics
- F1 score
- Accuracy
- Confusion matrices

## Documentation

Detailed documentation for each component:
- [RelationalToRDFAPI Documentation](src/RelationalToRDFAPI/README.md)
- [Graph Learning Demo Guide](src/graph-learning-demo/demo.ipynb)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- RelBench framework for relational deep learning
- Apache Jena Fuseki for RDF storage
- PyTorch Geometric for graph neural networks 