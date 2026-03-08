import pandas as pd
from google import genai
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
import hashlib
from .schema import EmailArtifact, Entity, Claim, Evidence
import time

class EnronExtractor:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.extraction_version = "1.0"
        self.schema_version = "1.0"
        
        # Define extraction prompt
        self.extraction_prompt = """
        You are analyzing Enron emails to build a memory graph of business activities.
        
        Email:
        From: {from_addr}
        To: {to_addrs}
        Subject: {subject}
        Date: {timestamp}
        
        Body:
        {body}
        
        Extract the following in JSON format:
        1. People mentioned (names, roles, relationships)
        2. Business activities (projects, meetings, decisions, approvals)
        3. Topics discussed
        4. Claims about entities (who works with whom, who approves what)
        5. Temporal information (deadlines, schedules, events)
        
        Rules:
        - Be conservative: only extract what's explicitly stated or strongly implied
        - For each extraction, provide:
          * The exact excerpt as evidence
          * Character offsets (approximate is fine)
          * Confidence score (0-1)
        - Normalize names (e.g., "Phillip Allen" -> "Phillip K Allen")
        
        Return JSON with this structure:
        {{
            "entities": [
                {{
                    "type": "person|team|project|topic",
                    "name": "canonical name",
                    "aliases": ["other names"],
                    "confidence": 0.95
                }}
            ],
            "claims": [
                {{
                    "type": "relationship|fact|event",
                    "subject": "entity name",
                    "predicate": "action/relation",
                    "object": "entity name or value",
                    "confidence": 0.9,
                    "excerpt": "supporting text"
                }}
            ],
            "topics": ["topic1", "topic2"]
        }}
        
        If nothing substantial to extract, return {{"entities": [], "claims": [], "topics": []}}
        """
        
    def parse_email(self, row: pd.Series) -> EmailArtifact:
        """Parse raw email from CSV"""
        file_path = row['file']
        raw_message = row['message']
        
        # Basic parsing - in production, use email.parser
        lines = raw_message.split('\n')
        headers = {}
        body_lines = []
        in_headers = True
        
        for line in lines:
            if in_headers and line.strip() == '':
                in_headers = False
                continue
            
            if in_headers and ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
            elif not in_headers:
                body_lines.append(line)
        
        # Parse addresses
        from_addr = headers.get('from', '')
        to_addrs_raw = headers.get('to', '')
        to_addrs = [addr.strip() for addr in to_addrs_raw.split(',') if addr.strip()]
        cc_addrs_raw = headers.get('cc', '')
        cc_addrs = [addr.strip() for addr in cc_addrs_raw.split(',') if addr.strip()]
        
        # Parse timestamp
        timestamp = None
        date_str = headers.get('date', '')
        try:
            # Try different date formats
            from email.utils import parsedate_to_datetime
            timestamp = parsedate_to_datetime(date_str)
        except:
            try:
                timestamp = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
            except:
                try:
                    timestamp = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
                except:
                    timestamp = datetime.now()
        
        return EmailArtifact(
            message_id=headers.get('message-id', '').strip('<>'),
            file_path=file_path,
            from_addr=from_addr,
            to_addrs=to_addrs,
            cc_addrs=cc_addrs,
            bcc_addrs=[],
            subject=headers.get('subject', ''),
            body='\n'.join(body_lines),
            timestamp=timestamp,
            x_from=headers.get('x-from', ''),
            x_to=headers.get('x-to', '')
        )
    
    def extract_from_email(self, email: EmailArtifact) -> Tuple[List[Entity], List[Claim], List[Evidence]]:
        """Use Gemini to extract entities and claims from an email"""
        
        # Prepare prompt
        prompt = self.extraction_prompt.format(
            from_addr=email.from_addr,
            to_addrs=', '.join(email.to_addrs),
            subject=email.subject,
            timestamp=email.timestamp.isoformat() if email.timestamp else 'Unknown',
            body=email.body[:2000]  # Limit length for API
        )
        
        # Call Gemini with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                # Parse JSON from response
                json_str = self._extract_json(response.text)
                if not json_str:
                    return [], [], []
                
                data = json.loads(json_str)
                
                # Convert to our schema
                entities = []
                claims = []
                evidences = []
                
                # Create evidence for this email
                email_evidence = Evidence(
                    source_id=email.file_path,
                    excerpt=email.body[:500],  # Truncate for storage
                    start_offset=0,
                    end_offset=min(500, len(email.body)),
                    timestamp=email.timestamp,
                    confidence=1.0
                )
                evidences.append(email_evidence)
                
                # Create entities
                entity_map = {}  # name -> entity_id
                for ent_data in data.get('entities', []):
                    entity = Entity(
                        type=ent_data['type'],
                        names=[ent_data['name']] + ent_data.get('aliases', []),
                        confidence=ent_data.get('confidence', 0.8),
                        evidence=[email_evidence]
                    )
                    entities.append(entity)
                    entity_map[ent_data['name']] = entity.entity_id
                
                # Create claims
                for claim_data in data.get('claims', []):
                    # Find subject entity
                    subject_id = None
                    for name, eid in entity_map.items():
                        if claim_data['subject'].lower() in name.lower():
                            subject_id = eid
                            break
                    
                    if not subject_id:
                        # Create placeholder entity
                        placeholder = Entity(
                            type='unknown',
                            names=[claim_data['subject']],
                            confidence=0.5,
                            evidence=[email_evidence]
                        )
                        entities.append(placeholder)
                        subject_id = placeholder.entity_id
                        entity_map[claim_data['subject']] = subject_id
                    
                    # Find object entity if exists
                    object_id = None
                    if 'object' in claim_data and claim_data['object']:
                        for name, eid in entity_map.items():
                            if claim_data['object'].lower() in name.lower():
                                object_id = eid
                                break
                        
                        if not object_id and claim_data['object']:
                            placeholder = Entity(
                                type='unknown',
                                names=[claim_data['object']],
                                confidence=0.5,
                                evidence=[email_evidence]
                            )
                            entities.append(placeholder)
                            object_id = placeholder.entity_id
                            entity_map[claim_data['object']] = object_id
                    
                    # Create claim evidence
                    excerpt = claim_data.get('excerpt', email.body[:200])
                    start_offset = email.body.find(excerpt) if excerpt in email.body else 0
                    
                    claim_evidence = Evidence(
                        source_id=email.file_path,
                        excerpt=excerpt,
                        start_offset=start_offset,
                        end_offset=start_offset + len(excerpt),
                        timestamp=email.timestamp,
                        confidence=claim_data.get('confidence', 0.8)
                    )
                    evidences.append(claim_evidence)
                    
                    claim = Claim(
                        type=claim_data.get('type', 'relationship'),
                        subject_id=subject_id,
                        predicate=claim_data['predicate'],
                        valid_from=email.timestamp,
                        object_id=object_id,
                        value=claim_data.get('object'),
                        confidence=claim_data.get('confidence', 0.8),
                        evidence=[claim_evidence]
                    )
                    claims.append(claim)
                
                return entities, claims, evidences
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return [], [], []
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from model response"""
        # Try to find JSON between triple backticks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1)
        
        # Try to find anything that looks like JSON
        json_match = re.search(r'({[\s\S]*})', text)
        if json_match:
            return json_match.group(1)
        
        return None