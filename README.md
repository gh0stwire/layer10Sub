# Enron Memory Graph: Layer10 Implementation

# Usage
python run_pipeline.py --csv data/raw/mails.csv --api-key (model is gemini 2.5 flash right now, so corresponding api key) --sample (N, the number of emails to process) <br>
// This is to make the json files. Then visualized using flask.<br>
python app.py <br>
// this is for the visualization. will run flas on port 5000 of 127.0.0.1<br>

# Enron Memory Graph

A grounded long-term memory system built on the Enron email corpus, demonstrating concepts for Layer10.

## Setup

1. Install dependencies:
pip install -r requirements.txt

## Overview
This implementation builds a grounded memory graph from the Enron email corpus, demonstrating key concepts for Layer10's long-term memory system.

## 1. Ontology Design

### Entity Types
- **Person**: Individuals in the organization (e.g., "Phillip Allen", "Tim Belden")
- **Team**: Groups/departments (e.g., "Trading", "Legal")
- **Project**: Initiatives/work items (e.g., "Forecast", "Business meeting")
- **Topic**: Discussion subjects (e.g., "Trading strategy", "Travel planning")

### Claim Types
- **works_with**: Collaboration relationships
- **discusses**: Topic associations
- **plans**: Scheduled activities
- **approves**: Decision authority
- **informs**: Information flow
- **schedules**: Temporal events

### Evidence Structure
Every claim is grounded with:
- Source ID (email file path)
- Exact excerpt
- Character offsets
- Timestamp
- Confidence score

## 2. Extraction Pipeline

### Schema Design
The extraction contract uses a flexible JSON schema that captures:
- Entities with canonical names and aliases
- Typed relationships with evidence
- Confidence scores for quality gates

### Validation & Repair
- Retry logic with exponential backoff for API calls
- JSON parsing with fallback strategies
- Confidence thresholds (0.7 minimum for inclusion)
- Cross-validation across multiple emails

### Versioning Strategy
- Extraction version tracked in graph metadata
- Schema version embedded in each extraction
- Backfill capability through reprocessing pipeline

## 3. Deduplication Strategy

### Artifact-Level (Emails)
- Fingerprint based on normalized body + sender + date
- Thread detection via In-Reply-To headers
- Quote stripping and signature removal

### Entity-Level (People/Projects)
- Name normalization (email→name, "Last, First"→"First Last")
- Similarity threshold (0.85) for merging
- Alias tracking and audit trail

### Claim-Level
- Merge identical claims from multiple sources
- Evidence aggregation
- Confidence boosting with multiple corroborations

### Conflict Resolution
- Temporal validity windows (valid_from/valid_to)
- Current state vs. historical records
- Support for decision reversals

## 4. Memory Graph Design

### Core Objects
```python
Entity: {id, type, names, properties, evidence}
Claim: {id, type, subject, predicate, object, validity, evidence}
Evidence: {source_id, excerpt, offsets, timestamp}
