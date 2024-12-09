# SAFIR: A Self-Adaptive Semantic Framework for Data Interoperability in IoT and Healthcare Systems
> **Bridging Semantic Interoperability and Adaptive Learning in Complex Systems**

## Research Overview
This repository presents a approach to enhancing self-adaptive systems (SAS) through semantic knowledge graph transformation and intelligent machine learning techniques. Our methodology addresses fundamental challenges in system adaptability by creating a flexible, semantically-rich framework for dynamic system reconfiguration.

### Research Contribution
Our work introduces an approach that:
- Enables semantic interoperability across heterogeneous data sources
- Develops a generalizable framework for knowledge graph construction
- Integrates machine learning for predictive system adaptation
- Demonstrates adaptive decision-making through semantic reasoning

---

## Theoretical Foundation

### Self-Adaptive Systems Challenges
Traditional self-adaptive systems struggle with:
- Rigid data representations
- Limited contextual understanding
- Inflexible adaptation mechanisms
- Difficulty integrating diverse information sources

### Proposed Solution
Our approach addresses these challenges through:
- **Semantic Transformation:** Converting heterogeneous data into a unified knowledge representation
- **Adaptive Learning:** Implementing graph-based machine learning for predictive insights
- **Dynamic Reconfiguration:** Enabling intelligent system responses based on learned patterns

---

## Technical Architecture

### Workflow Components
1. **Semantic Data Transformation**
   - Ontology-driven mapping of diverse data sources
   - Semantic enrichment using standard vocabularies
   - Preserving contextual information across transformations

2. **Knowledge Graph Construction**
   - Convert mapped data into structured graph representations
   - Apply semantic annotations and metadata
   - Create a flexible, interconnected knowledge base

3. **Adaptive Learning Mechanism**
   - Implement machine learning models on semantic graphs
   - Support predictive tasks:
     * Anomaly detection
     * System state prediction
     * Adaptive decision support

4. **Self-Adaptive System Integration**
   - Implement MAPE-K (Monitor-Analyze-Plan-Execute-Knowledge) loop
   - Enable dynamic system reconfiguration
   - Provide feedback-driven adaptation strategies

---

## Methodological Innovation

### Semantic Interoperability
- **Challenge:** Integrating diverse data sources with different semantics
- **Solution:** Develop a unified ontological framework
- **Approach:** 
  * Use standard ontologies (SOSA, FHIR, UCUM)
  * Create mappings that preserve semantic richness
  * Enable cross-domain knowledge integration

### Adaptive Learning
- **Challenge:** Creating intelligent, context-aware adaptation
- **Solution:** Graph-based machine learning
- **Approach:**
  * Represent system state as a dynamic knowledge graph
  * Use graph neural networks for predictive modeling
  * Enable continuous learning and adaptation

---

## Research Implications

### Theoretical Contributions
- Novel framework for semantic-driven self-adaptation
- Demonstration of graph-based learning in complex systems
- Extensible approach to system intelligence

### Practical Applications
- Intelligent systems with context-aware adaptation
- Enhanced decision support in:
  * Healthcare monitoring
  * IoT systems
  * Complex engineering environments

---

## Key Features
- **Data Transformation:**
  - Converts raw data into RDF using RML specifications.
  - Supports semantic enrichment with healthcare ontologies (e.g., SOSA, FHIR, UCUM).
- **Unified Knowledge Representation:**
  - Creates a single ontology unifying data from multiple pilot projects.
- **Graph Machine Learning Integration:**
  - Use of RDF as a graph structure for tasks like node classification, link prediction, and anomaly detection.
- **Self-Adaptive Systems Context:**
  - Demonstrates the use of RDF and Graph ML in monitoring, analysis, planning, and execution (MAPE-K loop).

---

## Repository Structure
- **`requirements/`:** Documentation for dependencies and setup.
- **`data/`:** Raw input data, RDF outputs, and graph representations.
- **`rml/`:** RML mapping scripts for standard, improved, and unified mappings.
- **`rdf/`:** RDF outputs generated from RML mappings.
- **`scripts/`:** Scripts for automating RDF generation and graph conversion.
- **`graph_ml/`:** Graph ML workflows, models, and visualizations.
- **`tests/`:** Test cases for reproducibility and validation.
- **`docs/`:** Documentation for the process and Graph ML integration.

---

## Installation
1. **Install Dependencies:**
   - Java 17
   - RMLMapper v6.5.1
   - Python 3.8+ with required libraries:
     ```bash
     pip install networkx torch dgl matplotlib
     ```

2. **Setup Environment:**
   - Follow `requirements/installation.md` for detailed instructions.

---

## Workflow
### 1. **Data Transformation**
Run RML scripts to generate RDF from raw data:
```bash
./scripts/generate_rdf.sh rml/standard/source_a.ttl rdf/standard/source_a.ttl
```

### 2. **Semantic Enrichment**
Use improved RML scripts with ontologies like SOSA and FHIR:
```bash
./scripts/generate_rdf.sh rml/improved/source_a.ttl rdf/improved/source_a.ttl
```

### 3. **Graph Construction**
Convert RDF data into a graph format:
```bash
python scripts/graph_converter.py --input rdf/improved/source_a.ttl --output data/graph/source_a.graphml
```

### 4. **Graph Machine Learning**
Run a Graph ML task (e.g., node classification):
```bash
python graph_ml/models/node_classification.py --graph data/graph/source_a.graphml
```

### 5. **Self-Adaptive System Simulation**
Integrate Graph ML outputs into the self-adaptive process (e.g., sending alerts based on predictions).

---

## Example Use Case: Healthcare Monitoring
- **Scenario:** Monitor patient health using IoT devices and environmental data.
- **Steps:**
  1. Transform sensor data to RDF.
  2. Enrich RDF with semantic annotations.
  3. Build a knowledge graph.
  4. Use Graph ML to detect anomalies or predict patient risk.
  5. Trigger adaptive actions (e.g., caregiver notifications, environment adjustments).

---

## Graph Machine Learning Tasks
- **Node Classification:** Predict missing properties (e.g., risk levels for patients).
- **Link Prediction:** Infer missing relationships (e.g., correlation between activities and health metrics).
- **Anomaly Detection:** Identify unusual patterns in data.

---

## Testing
- **RDF Validation:**
  ```bash
  ./tests/validation/validate_rdf.sh rdf/improved/source_a.ttl
  ```
- **Graph ML Testing:**
  ```bash
  python tests/graph_tests/test_node_classification.py
  ```

---

## Limitations and Future Work
- Current implementation is a proof-of-concept
- Future research directions:
  * Scalability improvements
  * Generalization across more domains
  * Advanced machine learning architectures

---

# References

## Mapping and Data Transformation

### Resource Description Mapping
- **RML (Resource Description Mapping Language)**
  - **Documentation**: [RML Official Documentation](https://rml.io/docs/)
  - Comprehensive guide for mapping heterogeneous data sources to RDF

- **R2RML and Mapping Extensions**
  - **Paper**: "R2RML-F: Towards Sharing and Executing Domain Logic in R2RML Mappings"
    - **Author**: Chr. De Bruyne et al.
    - **Link**: [Preprint PDF](https://chrdebru.github.io/papers/2016-ldow-preprint.pdf)
    - *Key Focus*: Extending R2RML mappings to support domain-specific logic and reusability

## Semantic and Ontological Resources

### Ontology Design
- **DOLCE Ultra Lite (DUL)**
  - **Ontology Link**: [DUL Ontology](http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#)
  - Foundational ontology for knowledge representation

### Semantic Sensor Networks
- **SOSA (Sensor, Observation, Sample, Actuator)**
  - **Namespace**: [SOSA Ontology](http://www.w3.org/ns/sosa)

- **SSN (Semantic Sensor Network)**
  - **Specification**: [SSN Ontology](https://www.w3.org/TR/vocab-ssn/)
  - Comprehensive ontology for describing sensors, observations, and related concepts

### Healthcare Interoperability
- **FHIR (Fast Healthcare Interoperability Resources)**
  - **RDF Specification**: [FHIR RDF Documentation](https://build.fhir.org/rdf.html)
  - Standard for exchanging healthcare data with RDF support

### Measurement and Standardization
- **UCUM (Unified Code for Units of Measure)**
  - **Data Reference**: [UCUM Data](https://download.hl7.de/documents/ucum/ucumdata.html)
  - Standardized representation of units and codes

---

## License
This repository is licensed under the [License](LICENSE).
