#!/usr/bin/env python3
"""
Main pipeline for Enron memory graph construction
"""

import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.extraction.extractor import EnronExtractor
from src.extraction.schema import EmailArtifact, Evidence
from src.deduplication.artifact import ArtifactDeduplicator
from src.graph.builder import MemoryGraph
from src.retrieval.search import MemoryRetriever

def load_emails(csv_path: str, sample_size: int = None):
    """Load emails from CSV"""
    print(f"Loading emails from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        
        if sample_size:
            df = df.head(sample_size)
        
        print(f"Loaded {len(df)} emails")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def parse_emails(df, extractor):
    """Parse raw emails into EmailArtifact objects"""
    emails = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing emails"):
        try:
            email = extractor.parse_email(row)
            emails.append(email)
        except Exception as e:
            print(f"Error parsing email {idx}: {e}")
    
    print(f"Successfully parsed {len(emails)} emails")
    return emails

def extract_from_emails(emails, extractor, max_emails: int = 100):
    """Extract entities and claims from emails"""
    all_entities = []
    all_claims = []
    all_evidence = []
    
    for i, email in enumerate(tqdm(emails[:max_emails], desc="Extracting from emails")):
        try:
            print(f"\nProcessing email {i+1}/{min(max_emails, len(emails))}: {email.file_path}")
            entities, claims, evidence = extractor.extract_from_email(email)
            all_entities.extend(entities)
            all_claims.extend(claims)
            all_evidence.extend(evidence)
            print(f"  Found {len(entities)} entities, {len(claims)} claims")
        except Exception as e:
            print(f"Error extracting from {email.file_path}: {e}")
    
    print(f"\nExtracted {len(all_entities)} entities, {len(all_claims)} claims total")
    return all_entities, all_claims, all_evidence

def deduplicate_artifacts(emails):
    """Deduplicate email artifacts"""
    deduplicator = ArtifactDeduplicator()
    unique_emails = deduplicator.deduplicate(emails)
    threads = deduplicator.group_by_thread(unique_emails)
    
    print(f"Deduplicated {len(emails)} -> {len(unique_emails)} unique emails")
    print(f"Grouped into {len(threads)} conversation threads")
    
    return unique_emails, threads

def save_outputs(graph: MemoryGraph, retriever: MemoryRetriever, output_dir: str):
    """Save graph and example contexts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graph
    graph_export = graph.export_graph()
    with open(os.path.join(output_dir, 'graph.json'), 'w') as f:
        json.dump(graph_export, f, indent=2, default=str)
    
    # Generate and save example contexts
    example_queries = [
        "Who works on trading?",
        "What meetings are being planned?",
        "Who approves deals?",
        "What projects are discussed?"
    ]
    
    contexts = []
    for query in example_queries:
        try:
            context = retriever.retrieve_context(query)
            contexts.append({
                'query': query,
                'context': context,
                'formatted': retriever.format_citations(context)
            })
        except Exception as e:
            print(f"Error retrieving context for '{query}': {e}")
    
    with open(os.path.join(output_dir, 'contexts.json'), 'w') as f:
        json.dump(contexts, f, indent=2, default=str)
    
    # Save summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Enron Memory Graph Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Entities: {len(graph.entities)}\n")
        f.write(f"Claims: {len(graph.claims)}\n")
        f.write(f"Evidence items: {len(graph.evidence_store)}\n")
        
        f.write("\nEntity Types:\n")
        entity_types = {}
        for entity in graph.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        for etype, count in entity_types.items():
            f.write(f"  {etype}: {count}\n")
        
        f.write("\nExample Contexts:\n")
        for ctx in contexts:
            f.write(f"\nQuery: {ctx['query']}\n")
            f.write(f"{ctx['formatted']}\n")
    
    print(f"Outputs saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Build Enron memory graph')
    parser.add_argument('--csv', type=str, default='data/raw/mails.csv',
                        help='Path to Enron CSV file')
    parser.add_argument('--api-key', type=str, required=True,
                        help='Gemini API key')
    parser.add_argument('--sample', type=int, default=10,
                        help='Number of emails to process (for testing)')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Enron Memory Graph Builder")
    print("="*60)
    print(f"API Key: {args.api_key[:5]}...{args.api_key[-5:]}")
    print(f"Sample size: {args.sample}")
    print(f"Output dir: {args.output}")
    print("="*60)
    
    # Initialize components
    try:
        extractor = EnronExtractor(api_key=args.api_key)
        print("✓ Extractor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize extractor: {e}")
        return
    
    graph = MemoryGraph()
    print("✓ Graph initialized")
    
    # Load emails
    df = load_emails(args.csv, args.sample)
    if df is None:
        return
    
    # Parse emails
    emails = parse_emails(df, extractor)
    if not emails:
        print("No emails parsed successfully")
        return
    
    # Deduplicate artifacts
    unique_emails, threads = deduplicate_artifacts(emails)
    
    # Extract from emails
    all_entities, all_claims, all_evidence = extract_from_emails(
        unique_emails, extractor, max_emails=min(args.sample, len(unique_emails))
    )
    
    # Add to graph
    for email in unique_emails[:min(args.sample, len(unique_emails))]:
        graph.add_extraction(all_entities, all_claims, all_evidence, email.file_path)
    
    # Initialize retriever
    retriever = MemoryRetriever(graph)
    
    # Save outputs
    save_outputs(graph, retriever, args.output)
    
    print("\n✓ Pipeline completed successfully!")
    print(f"To start visualization:")
    print(f"  cd src/visualization")
    print(f"  python app.py")
    print(f"  Then open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()
