from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
from ..graph.builder import MemoryGraph
from ..extraction.schema import Claim, Entity, Evidence

class MemoryRetriever:
    """Retrieve grounded context from memory graph"""
    
    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self.embedding_cache = {}
        
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Simple keyword-based search"""
        query_terms = set(query.lower().split())
        scores = []
        
        # Search entities
        for entity_id, entity in self.graph.entities.items():
            entity_text = ' '.join(entity.names).lower()
            matches = sum(1 for term in query_terms if term in entity_text)
            if matches > 0:
                scores.append((entity_id, matches / len(query_terms), 'entity'))
        
        # Search claims
        for claim_id, claim in self.graph.claims.items():
            claim_text = f"{claim.predicate} {claim.value or ''}".lower()
            matches = sum(1 for term in query_terms if term in claim_text)
            if matches > 0:
                scores.append((claim_id, matches / len(query_terms), 'claim'))
        
        # Sort and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(item_id, score) for item_id, score, _ in scores[:top_k]]
    
    def expand_entities(self, entity_ids: List[str], 
                        max_depth: int = 2,
                        max_claims: int = 20) -> List[Claim]:
        """Expand from entities to connected claims"""
        visited_entities = set(entity_ids)
        all_claims = []
        current_depth = 0
        
        while current_depth < max_depth and len(all_claims) < max_claims:
            new_entities = set()
            
            for entity_id in list(visited_entities):
                # Get claims where entity is subject
                subject_claims = self.graph.claims_by_subject.get(entity_id, [])
                for claim_id in subject_claims:
                    claim = self.graph.claims.get(claim_id)
                    if claim and claim.is_current and claim_id not in [c.claim_id for c in all_claims]:
                        all_claims.append(claim)
                        if claim.object_id and claim.object_id not in visited_entities:
                            new_entities.add(claim.object_id)
                
                # Get claims where entity is object
                object_claims = self.graph.claims_by_object.get(entity_id, [])
                for claim_id in object_claims:
                    claim = self.graph.claims.get(claim_id)
                    if claim and claim.is_current and claim_id not in [c.claim_id for c in all_claims]:
                        all_claims.append(claim)
                        if claim.subject_id not in visited_entities:
                            new_entities.add(claim.subject_id)
            
            visited_entities.update(new_entities)
            current_depth += 1
            
            if len(all_claims) >= max_claims:
                break
        
        return all_claims[:max_claims]
    
    def rank_claims(self, claims: List[Claim], query: str) -> List[Tuple[Claim, float]]:
        """Rank claims by relevance to query"""
        query_terms = set(query.lower().split())
        scored_claims = []
        
        for claim in claims:
            score = 0.0
            
            # Term matching
            claim_text = f"{claim.predicate} {claim.value or ''}".lower()
            term_matches = sum(1 for term in query_terms if term in claim_text)
            score += term_matches * 0.3
            
            # Recency
            if claim.valid_from:
                days_old = (datetime.now() - claim.valid_from).days
                recency_score = max(0, 1 - (days_old / 365))  # Decay over a year
                score += recency_score * 0.2
            else:
                score += 0.1  # Low score if no date
            
            # Confidence
            score += claim.confidence * 0.3
            
            # Evidence strength
            evidence_score = min(1.0, len(claim.evidence) / 3)
            score += evidence_score * 0.2
            
            scored_claims.append((claim, score))
        
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        return scored_claims
    
    def retrieve_context(self, query: str, 
                        max_claims: int = 10,
                        include_evidence: bool = True) -> Dict[str, Any]:
        """Main retrieval API"""
        
        # Step 1: Initial search
        initial_results = self.keyword_search(query, top_k=20)
        
        # Step 2: Separate entities and claims
        entity_ids = [item_id for item_id, _ in initial_results 
                     if item_id in self.graph.entities]
        claim_ids = [item_id for item_id, _ in initial_results 
                    if item_id in self.graph.claims]
        
        # Step 3: Expand from entities
        expanded_claims = self.expand_entities(entity_ids, max_depth=2)
        
        # Step 4: Add directly matched claims
        all_claims = expanded_claims + [self.graph.claims[cid] 
                                        for cid in claim_ids if cid in self.graph.claims]
        
        # Step 5: Remove duplicates
        seen = set()
        unique_claims = []
        for claim in all_claims:
            if claim.claim_id not in seen:
                seen.add(claim.claim_id)
                unique_claims.append(claim)
        
        # Step 6: Rank
        ranked = self.rank_claims(unique_claims, query)
        
        # Step 7: Build context pack
        context = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'claims': [],
            'evidence': [],
            'entities': {}
        }
        
        for claim, score in ranked[:max_claims]:
            # Get subject entity
            subject = self.graph.entities.get(claim.subject_id)
            subject_name = subject.names[0] if subject else 'Unknown'
            
            # Get object entity
            object_name = None
            if claim.object_id:
                obj = self.graph.entities.get(claim.object_id)
                object_name = obj.names[0] if obj else claim.value
            
            claim_data = {
                'id': claim.claim_id,
                'type': claim.type,
                'subject': subject_name,
                'predicate': claim.predicate,
                'object': object_name,
                'value': claim.value,
                'confidence': claim.confidence,
                'relevance_score': score,
                'valid_from': claim.valid_from.isoformat() if claim.valid_from else None,
                'evidence_ids': [e.fingerprint for e in claim.evidence]
            }
            context['claims'].append(claim_data)
            
            # Add entities to context
            if subject and subject.entity_id not in context['entities']:
                context['entities'][subject.entity_id] = {
                    'id': subject.entity_id,
                    'type': subject.type,
                    'names': subject.names,
                    'confidence': subject.confidence
                }
            
            if claim.object_id and claim.object_id in self.graph.entities:
                obj = self.graph.entities[claim.object_id]
                if obj.entity_id not in context['entities']:
                    context['entities'][obj.entity_id] = {
                        'id': obj.entity_id,
                        'type': obj.type,
                        'names': obj.names,
                        'confidence': obj.confidence
                    }
            
            # Add evidence
            if include_evidence:
                for evidence in claim.evidence:
                    if evidence.fingerprint not in [e['id'] for e in context['evidence']]:
                        context['evidence'].append({
                            'id': evidence.fingerprint,
                            'source_id': evidence.source_id,
                            'excerpt': evidence.excerpt,
                            'timestamp': evidence.timestamp.isoformat() if evidence.timestamp else None,
                            'confidence': evidence.confidence
                        })
        
        return context
    
    def format_citations(self, context: Dict[str, Any]) -> str:
        """Format context with citations for LLM consumption"""
        output = []
        output.append(f"Query: {context['query']}\n")
        output.append("Retrieved Information:\n")
        
        for i, claim in enumerate(context['claims'], 1):
            output.append(f"\n[{i}] {claim['subject']} {claim['predicate']} {claim['object'] or claim['value'] or ''}")
            output.append(f"    Confidence: {claim['confidence']:.2f}, Relevance: {claim['relevance_score']:.2f}")
            if claim['valid_from']:
                output.append(f"    Date: {claim['valid_from']}")
            
            # Add evidence excerpts
            for j, evidence_id in enumerate(claim['evidence_ids'][:2]):  # Limit to 2 per claim
                evidence = next((e for e in context['evidence'] if e['id'] == evidence_id), None)
                if evidence:
                    output.append(f"    Evidence {j+1}: \"{evidence['excerpt'][:100]}...\"")
                    output.append(f"    Source: {evidence['source_id']}")
        
        return '\n'.join(output)