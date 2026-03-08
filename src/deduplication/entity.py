from typing import List, Dict, Set, Optional
import re
from difflib import SequenceMatcher
from ..extraction.schema import Entity, Evidence
from datetime import datetime

class EntityCanonicalizer:
    """Canonicalize entities (people, teams, projects)"""
    
    def __init__(self):
        self.canonical_map: Dict[str, str] = {}  # alias -> canonical_id
        self.entity_clusters: Dict[str, List[Entity]] = {}  # canonical_id -> entities
        self.merge_audit: List[Dict] = []  # Audit trail for reversibility
        
    def normalize_name(self, name: str) -> str:
        """Normalize person/entity names"""
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Handle email addresses
        email_match = re.match(r'([^@]+)@', name)
        if email_match:
            name = email_match.group(1).replace('.', ' ')
        
        # Handle "Last, First" format
        if ',' in name and not '@' in name:
            parts = [p.strip() for p in name.split(',')]
            if len(parts) == 2:
                name = f"{parts[1]} {parts[0]}"
        
        # Remove common prefixes
        name = re.sub(r'^(Dr|Mr|Mrs|Ms|Prof)\.?\s+', '', name, flags=re.IGNORECASE)
        
        return name.lower().strip()
    
    def name_similarity(self, name1: str, name2: str) -> float:
        """Compute similarity between two names"""
        n1 = self.normalize_name(name1)
        n2 = self.normalize_name(name2)
        
        if n1 == n2:
            return 1.0
        
        # Check if one is a subset of the other
        if n1 in n2 or n2 in n1:
            return 0.9
        
        # Use sequence matching
        return SequenceMatcher(None, n1, n2).ratio()
    
    def find_canonical_entity(self, entity: Entity, threshold: float = 0.85) -> Optional[str]:
        """Find if entity matches an existing canonical entity"""
        for name in entity.names:
            for canonical_id, cluster in self.entity_clusters.items():
                for existing in cluster:
                    for existing_name in existing.names:
                        similarity = self.name_similarity(name, existing_name)
                        if similarity >= threshold:
                            return canonical_id
        return None
    
    def canonicalize(self, entities: List[Entity]) -> List[Entity]:
        """Canonicalize entities, merging similar ones"""
        canonical_entities = []
        
        for entity in entities:
            # Try to find matching canonical entity
            canonical_id = self.find_canonical_entity(entity)
            
            if canonical_id:
                # Merge into existing cluster
                self.merge_audit.append({
                    'action': 'merge',
                    'source_id': entity.entity_id,
                    'target_id': canonical_id,
                    'reason': 'name_similarity',
                    'names': entity.names
                })
                
                # Update existing cluster
                cluster = self.entity_clusters[canonical_id]
                cluster.append(entity)
                
                # Update canonical entity with new info
                canonical = cluster[0]  # First entity is canonical
                canonical.names = list(set(canonical.names + entity.names))
                canonical.confidence = max(canonical.confidence, entity.confidence)
                canonical.evidence.extend(entity.evidence)
                canonical.updated_at = max(canonical.updated_at or entity.created_at, 
                                         entity.created_at)
                
                # Update alias map
                for name in entity.names:
                    self.canonical_map[self.normalize_name(name)] = canonical_id
            else:
                # Create new canonical cluster
                canonical_id = entity.entity_id
                self.entity_clusters[canonical_id] = [entity]
                canonical_entities.append(entity)
                
                for name in entity.names:
                    self.canonical_map[self.normalize_name(name)] = canonical_id
        
        return canonical_entities
    
    def get_merge_audit(self, entity_id: str) -> List[Dict]:
        """Get audit trail for a specific entity"""
        return [entry for entry in self.merge_audit 
                if entry['source_id'] == entity_id or entry['target_id'] == entity_id]
    
    def undo_merge(self, entity_id: str) -> Optional[List[Entity]]:
        """Undo a merge operation (reversibility)"""
        # Find merges involving this entity
        merges = [entry for entry in self.merge_audit 
                  if entry['target_id'] == entity_id]
        
        if not merges:
            return None
        
        # Get original entities
        original_entities = []
        for merge in merges:
            # Remove from cluster
            cluster = self.entity_clusters[entity_id]
            original = next((e for e in cluster if e.entity_id == merge['source_id']), None)
            if original:
                cluster.remove(original)
                original_entities.append(original)
                
                # Remove from alias map
                for name in original.names:
                    del self.canonical_map[self.normalize_name(name)]
        
        return original_entities