# deep-researcher

# Topic: Deep Researcher Cloud-Based Intelligent Research Assistant

## Project Overview

Deep Researcher is a cloud-native, AI-powered research assistant that automates the process of web-based academic and technical research. Utilizing a fine-tuned **LLaMA-3.2-1B Instruct model**, it iteratively refines queries, identifies knowledge gaps, and generates structured markdown reports with cited sources.

The project is deployed using **AWS ECS and Fargate**, providing a scalable, secure, and serverless infrastructure. Docker containers are used for consistent environments, with Amazon ECR handling image versioning. The system ensures dynamic research cycles, seamless updates, and efficient resource management, making it an ideal tool for deep, continuous research workflows.

## Team Members
1. Ayush Singh - 22bds012
2. Yashraj Kadam - 22bds066
3. Parishri Shah - 22bds043
4. Harsh Raj - 22bds027
5. Arya Raj - 22bds007


## Methodology

The system operates through a modular, iterative pipeline:

Frontend Interface: Users input a research topic and choose the refinement depth.

Query Generation: The fine-tuned LLaMA model generates context-aware search queries.

Web Scraping & Data Collection: Relevant content is extracted from reliable online sources.

Summarization & Gap Analysis: Summarizes content and identifies missing information.

Iterative Refinement: Refines queries and updates summaries until knowledge gaps are resolved.

Final Report Generation: Compiles the final output into a markdown file with citations.

The entire system is containerized and deployed via AWS ECS + Fargate, with traffic managed by an AWS Load Balancer and secure access via IAM roles, Security Groups, and Firewall rules.
<!-- ![Flow diagram](images/flowchart-2.jpg) -->

## How to run
1. Prerequisites<br />
a) AWS CLI, Docker, and ECS CLI installed<br />
b) An active Amazon ECR repository and ECS cluster<br />
c) Sufficient compute resources (for model size ~15GB)<br />

2.  Model Setup (LLaMA-3B)
The fine-tuned LLaMA-3B Instruct model (~15GB) is not included in this repository due to size constraints.
You have two options to set it up:
Option 1: Manual Download
a) Download the model files<br />
b) Place the files inside a folder named models/llama-3b/ in your local project directory<br />
c) Mount this directory into the Docker container during runtime<br />

Option 2: Auto-Download During Docker Build
If you've configured the Dockerfile to automatically download the model:
a) Ensure the script in the Docker container fetches the model from a secure, accessible URL<br />
b) Confirm the download path matches what the application expects<br />


## Presentation Slides
[PPT](Presentation-slides.pdf) 
