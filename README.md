# SproutML

A data science toolbox that enables expedient bootstrapping of new ML projects.

## The Pitch

For the Data Scientists and DevOps engineers that participate in data science focused Tech Challenges who need to demonstrate a deployable solution in a short amount of time, **SproutML** is a data science toolbox that allows you to quickly bootstrap a deployable, scalable supervised learning model.

Unlike SaaS offerings like CloudML or Python libraries like `sklearn`, **SproutML** lets you rapidly deliver a supervised learning model that’s deployed in your own cloud infrastructure, giving you the feature readiness of a SaaS offering, while letting you retain complete control over your workflow and data.

## Goal & Description

### Explore various technologies to build a reusable production infrastructure as a collaborative effort with DevOps and Data Science

* The focus is Natural Language Processing and classification of text documents
* The Doc2Vec unsupervised algorithm is used to create text vectors and associated subject tags to build a Logistic Regression classification model
* A flask app serves as the prediction endpoint endpoint
* The entire pipeline process has been containerized and deployed using Docker

### Prerequisites

1 - Python 3.6+

To install main dependencies, just run ```pip install -r requirements.txt```
All other packages used are available individually via, pip install

2 - Docker / DockerToolBox

## Running Python
The primary file that calls each of the functions is titled **pipeline.py**
This will print each stage of the pipeline in the console, as it completes each step of the process

The final output is a classification accuracy score for predicted vs. actual values

## Deployment

Add notes about Docker and Kubernetes deployment

## Team Contributors 
* **Ike Kramer** - *Data Science*
* **Dan Luhring** - *DevOps*
* **Cody Mitchell** - *Data Science*
* **George Paci** - *DevOps*
* **Karan Patel** - *Data Science*
* **Mary Scott Sanders** - *Data Science*
