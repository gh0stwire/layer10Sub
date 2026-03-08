from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import json
from collections import defaultdict
from ..extraction.schema import Entity, Claim, Evidence
from ..deduplication.entity import EntityCanonicalizer

class MemoryGraph:
    """Graph-based long-term memory store"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.claims: Dict[str, Claim] = {}
        self.evidence_store: Dict[str, Evidence] = {}
        self.entity_canonicalizer = EntityCanonicalizer()
        
        # Indexes
        self.entity_by_name: Dict[str, str] = {}  # name -> entity_id
        self.claims_by_subject: Dict[str, List[str]] = defaultdict(list)
        self.claims_by_object: Dict[str, List[str]] = defaultdict(list)
        self.evidence_by_source: Dict[str, List[str]] = defaultdict(list)
        
        # Versioning
        self.graph_version = "1.0"
        self.last_updated = datetime.now()
        self.update_log: List[Dict] = []
    
    def add_extraction(self, entities: List[Entity], claims: List[Claim], 
                       evidences: List[Evidence], source_id: str):
        """Add new extraction results to the graph"""
        
        # Store evidence first
        for evidence in evidences:
            evidence_id = evidence.fingerprint
            if evidence_id not in self.evidence_store:
                self.evidence_store[evidence_id] = evidence
                self.evidence_by_source[evidence.source_id].append(evidence_id)
        
        # Canonicalize and add entities
        canonical_entities = self.entity_canonicalizer.canonicalize(entities)
        for entity in canonical_entities:
            if entity.entity_id not in self.entities:
                self.entities[entity.entity_id] = entity
                # Index names
                for name in entity.names:
                    self.entity_by_name[name.lower()] = entity.entity_id
        
        # Add claims
        for claim in claims:
            if claim.claim_id not in self.claims:
                self.claims[claim.claim_id] = claim
                # Update indexes
                self.claims_by_subject[claim.subject_id].append(claim.claim_id)
                if claim.object_id:
                    self.claims_by_object[claim.object_id].append(claim.claim_id)
        
        # Log update
        self.update_log.append({
            'timestamp': datetime.now().isoformat(),
            'source_id': source_id,
            'entities_added': len([e for e in entities if e.entity_id not in self.entities]),
            'claims_added': len([c for c in claims if c.claim_id not in self.claims])
        })
        
        self.last_updated = datetime.now()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the graph"""
        current_entities = []
        current_claims = []
        
        for claim in self.claims.values():
            if claim.is_current:
                current_claims.append(claim)
        
        # Only include entities that have current claims or are recent
        entity_ids_with_claims = set(c.subject_id for c in current_claims) | \
                                 set(c.object_id for c in current_claims if c.object_id)
        
        for entity_id in entity_ids_with_claims:
            if entity_id in self.entities:
                current_entities.append(self.entities[entity_id])
        
        return {
            'entities': current_entities,
            'claims': current_claims,
            'timestamp': datetime.now().isoformat()
        }
    
    def handle_deletion(self, source_id: str):
        """Handle deletion of a source (email)"""
        # Find evidence from this source
        evidence_ids = self.evidence_by_source.get(source_id, [])
        
        # Find claims that depend on this evidence
        affected_claims = set()
        for evidence_id in evidence_ids:
            for claim_id, claim in self.claims.items():
                if any(e.fingerprint == evidence_id for e in claim.evidence):
                    affected_claims.add(claim_id)
        
        # Mark claims as no longer current if all evidence is gone
        for claim_id in affected_claims:
            claim = self.claims[claim_id]
            # Check if claim has other evidence
            other_evidence = [e for e in claim.evidence 
                            if e.fingerprint not in evidence_ids]
            if not other_evidence:
                claim.valid_to = datetime.now()  # Expire the claim
        
        # Remove evidence
        for evidence_id in evidence_ids:
            if evidence_id in self.evidence_store:
                del self.evidence_store[evidence_id]
        if source_id in self.evidence_by_source:
            del self.evidence_by_source[source_id]
    
    def handle_edit(self, source_id: str, new_evidence: Evidence):
        """Handle editing of a source"""
        # First handle as deletion
        self.handle_deletion(source_id)
        
        # Then add new evidence and re-extract (handled by pipeline)
        # This method just updates the evidence store
        evidence_id = new_evidence.fingerprint
        self.evidence_store[evidence_id] = new_evidence
        self.evidence_by_source[source_id].append(evidence_id)
    
    def export_graph(self) -> Dict[str, Any]:
        """Export graph for visualization"""
        return {
            'version': self.graph_version,
            'generated': datetime.now().isoformat(),
            'entities': [
                {
                    'id': e.entity_id,
                    'type': e.type,
                    'names': e.names,
                    'properties': e.properties,
                    'confidence': e.confidence,
                    'evidence_count': len(e.evidence)
                }
                for e in self.entities.values()
            ],
            'claims': [
                {
                    'id': c.claim_id,
                    'type': c.type,
                    'subject_id': c.subject_id,
                    'object_id': c.object_id,
                    'predicate': c.predicate,
                    'value': c.value,
                    'valid_from': c.valid_from.isoformat() if c.valid_from else None,
                    'valid_to': c.valid_to.isoformat() if c.valid_to else None,
                    'confidence': c.confidence,
                    'evidence_ids': [e.fingerprint for e in c.evidence]
                }
                for c in self.claims.values()
            ],
            'evidence': [
                {
                    'id': e.fingerprint,
                    'source_id': e.source_id,
                    'excerpt': e.excerpt[:200],  # Truncate for export
                    'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                    'confidence': e.confidence
                }
                for e in self.evidence_store.values()
            ]
        }