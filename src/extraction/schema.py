from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import hashlib

@dataclass
class Evidence:
    """Grounding evidence for any extracted claim"""
    source_id: str
    excerpt: str
    start_offset: int
    end_offset: int
    timestamp: datetime
    confidence: float = 1.0
    
    @property
    def fingerprint(self) -> str:
        """Unique identifier for this evidence"""
        content = f"{self.source_id}:{self.start_offset}:{self.end_offset}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

@dataclass
class Entity:
    """Core entity types in our memory graph"""
    type: str  # person, team, project, topic, company - MUST come first since it has no default
    names: List[str] = field(default_factory=list)
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    confidence: float = 1.0
    evidence: List[Evidence] = field(default_factory=list)

@dataclass
class Claim:
    """Relationships and facts about entities"""
    type: str  # works_with, discusses, approves, plans, schedules, etc. - MUST come first
    subject_id: str
    predicate: str
    valid_from: datetime
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    object_id: Optional[str] = None
    value: Any = None
    properties: Dict[str, Any] = field(default_factory=dict)
    valid_to: Optional[datetime] = None
    confidence: float = 1.0
    evidence: List[Evidence] = field(default_factory=list)
    
    @property
    def is_current(self) -> bool:
        """Check if claim is currently valid"""
        return self.valid_to is None or self.valid_to > datetime.now()

@dataclass
class EmailArtifact:
    """Represents an email as a source artifact"""
    message_id: str
    file_path: str
    from_addr: str
    timestamp: datetime
    body: str
    to_addrs: List[str] = field(default_factory=list)
    cc_addrs: List[str] = field(default_factory=list)
    bcc_addrs: List[str] = field(default_factory=list)
    subject: str = ""
    x_from: Optional[str] = None
    x_to: Optional[str] = None
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    @property
    def fingerprint(self) -> str:
        """Deduplication key"""
        # Normalize by removing quoting and whitespace
        import re
        normalized_body = ' '.join(self.body.strip().split())
        # Simple quote removal
        normalized_body = re.sub(r'>.*\n', '', normalized_body)
        content = f"{self.from_addr}:{self.timestamp}:{normalized_body[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()