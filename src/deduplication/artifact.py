from typing import List, Dict, Set, Optional
import hashlib
import re
from ..extraction.schema import EmailArtifact, Evidence

class ArtifactDeduplicator:
    """Deduplicate near-identical emails"""
    
    def __init__(self):
        self.fingerprint_map: Dict[str, List[EmailArtifact]] = {}
        self.thread_map: Dict[str, List[EmailArtifact]] = {}
        
    def normalize_body(self, body: str) -> str:
        """Remove quoting, signatures, and normalize whitespace"""
        # Remove quoted text (lines starting with >)
        lines = body.split('\n')
        filtered_lines = []
        
        for line in lines:
            if not line.startswith('>') and not line.startswith('On ') and 'wrote:' not in line:
                filtered_lines.append(line.strip())
        
        # Remove signatures (common patterns)
        body = '\n'.join(filtered_lines)
        body = re.sub(r'--+\s*\n.*$', '', body, flags=re.DOTALL)
        body = re.sub(r'_{10,}.*$', '', body, flags=re.DOTALL)
        
        # Normalize whitespace
        body = ' '.join(body.split())
        
        return body.lower().strip()
    
    def compute_fingerprint(self, email: EmailArtifact) -> str:
        """Compute deduplication fingerprint"""
        normalized_body = self.normalize_body(email.body)
        
        # Use from_addr, timestamp rounded to day, and body hash
        content = f"{email.from_addr}:{email.timestamp.date()}:{normalized_body[:500]}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def extract_thread_id(self, email: EmailArtifact) -> Optional[str]:
        """Extract or infer thread ID"""
        if hasattr(email, 'in_reply_to') and email.in_reply_to:
            return email.in_reply_to
        
        # Use subject as thread hint (remove Re:, Fwd:)
        subject = re.sub(r'^(Re|Fwd|FW|Fw):\s*', '', email.subject, flags=re.IGNORECASE)
        if subject:
            thread_hint = f"{subject}:{email.timestamp.date()}"
            return hashlib.sha256(thread_hint.encode()).hexdigest()
        
        return None
    
    def deduplicate(self, emails: List[EmailArtifact]) -> List[EmailArtifact]:
        """Return deduplicated list with merged metadata"""
        unique_emails = []
        
        for email in emails:
            fp = self.compute_fingerprint(email)
            
            if fp not in self.fingerprint_map:
                self.fingerprint_map[fp] = [email]
                unique_emails.append(email)
            else:
                # Merge metadata from duplicates
                existing = self.fingerprint_map[fp][0]
                # Keep the earliest timestamp
                if email.timestamp and existing.timestamp:
                    if email.timestamp < existing.timestamp:
                        existing.timestamp = email.timestamp
                # Merge recipients
                existing.to_addrs = list(set(existing.to_addrs + email.to_addrs))
                
        return unique_emails
    
    def group_by_thread(self, emails: List[EmailArtifact]) -> Dict[str, List[EmailArtifact]]:
        """Group emails into conversation threads"""
        threads = {}
        
        for email in emails:
            thread_id = self.extract_thread_id(email)
            if thread_id:
                if thread_id not in threads:
                    threads[thread_id] = []
                threads[thread_id].append(email)
        
        # Sort each thread by timestamp
        for thread_id in threads:
            threads[thread_id].sort(key=lambda x: x.timestamp if x.timestamp else datetime.min)
        
        return threads