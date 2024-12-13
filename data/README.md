# README for Real-World Data Simulation Component

## Overview

This project aims to address the need for generating additional input data for training and continuously updating a model with real-world data. The primary goal is to simulate a dynamic environment by generating diverse RDF data points, focusing on devices, sensors, events, and users. This enables the model to train on a broader dataset and better reflect real-world scenarios.

## Features

1. **Random RDF Data Generation**:
   - Simulates data for devices, sensors, events, and users.
   - Includes random readings for temperature, humidity, and other relevant sensor data points.
   - Creates RDF triples for a semantic representation of the simulated environment.

2. **Continuous Model Update**:
   - A placeholder for integrating the generated RDF data into a pipeline for model training.
   - Future enhancements will include real-time data injection into the model's training pipeline.

## Prerequisites

- Python 3.7 or later
- Required Python libraries: `random`, `datetime`
- RDF data handling tools (e.g., Apache Jena or similar) for processing the output RDF triples.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rdf-data-simulation.git
   cd rdf-data-simulation
   ```

2. Install dependencies (if additional libraries are added in the future):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Generate RDF Data**:
   Run the Python script to generate RDF data. By default, it creates 100 random sensor events.
   ```bash
   python generate_rdf_data.py
   ```

2. **Customize Data Generation**:
   You can modify the number of events or devices/sensors directly in the script by updating the `generate_rdf_data()` function:
   ```python
   new_rdf_data = generate_rdf_data(num_events=200)  # Generates 200 events
   ```

3. **View Generated Data**:
   The RDF data will be printed to the console. Redirect it to a file for further use:
   ```bash
   python generate_rdf_data.py > output_data.rdf
   ```

## File Structure

```
rdf-data-simulation/
├── generate_rdf_data.py   # Main script for data generation
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies (if applicable)
└── example_output.rdf     # Example RDF output
```

## Example RDF Output

Here is a snippet of the generated RDF data:

```turtle
@prefix ex: <http://example.org/> .

ex:dev_1 ex:hasDeviceName "Device dev_1" .
ex:dev_1 ex:hasOwner ex:usr_001 .
ex:sens_1 ex:hasSensorType "TEMPERATURE" .
ex:sens_1 ex:hasLocation "Location 2" .

ex:evt_1234 ex:timestamp "2024-12-13T12:00:00" ;
    ex:id "evt_1234" ;
    ex:userId "usr_3" ;
    ex:assetId "asset_456" ;
    ex:dimension "VALUE" ;
    ex:value 35.67 ;
    ex:originId "sens_1" ;
    ex:originType "TEMPERATURE" .
```

## Next Steps

- **Integration**: Feed the generated RDF data into the model training pipeline.
- **Data Validation**: Add RDF validation rules to ensure semantic consistency.
