#!/bin/bash

# Clone the repository
git clone https://github.com/HULTIG/safir-self-adaptive-semantic-framework.git
cd safir-self-adaptive-semantic-framework

# Download the RMLMapper JAR
wget https://github.com/RMLio/rmlmapper-java/releases/download/v6.5.1/rmlmapper-6.5.1-r371-all.jar

# Create the necessary directory
mkdir -p src/RelationalToRDFAPI/src/main/resources/jar/

# Move the JAR to the appropriate directory
mv rmlmapper-6.5.1-r371-all.jar src/RelationalToRDFAPI/src/main/resources/jar/

# Start the environment - it will take some time to build the services for the first time due to dependencies
docker-compose -f src/docker-compose.yml up -d --build

# Access the Spring Boot API
echo "Access the Spring Boot API at: http://localhost:8080/swagger-ui.html"

# Access the Jupyter Notebook by copying the token from the terminal output of the notebook service
echo "Access the Jupyter Notebook by copying the token from the terminal output of the notebook service:"

docker ps --filter "name=notebook" --format "{{.ID}}" | xargs -I {} docker logs {} | grep ?token