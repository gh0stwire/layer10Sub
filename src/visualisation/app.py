from flask import Flask, render_template, jsonify, request
import json
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

app = Flask(__name__)

# Load graph from saved file
graph_path = os.path.join(project_root, 'outputs', 'graph.json')
print(f"📊 Loading graph from: {graph_path}")

def load_saved_graph():
    if not os.path.exists(graph_path):
        print(f"❌ Graph file not found at: {graph_path}")
        return {'entities': [], 'claims': [], 'evidence': []}
    
    try:
        with open(graph_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data.get('entities', []))} entities and {len(data.get('claims', []))} claims")
        return data
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return {'entities': [], 'claims': [], 'evidence': []}

graph_data = load_saved_graph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graph')
def get_graph():
    """Return graph for visualization with validation"""
    print(f"📊 Building graph from {len(graph_data['entities'])} entities and {len(graph_data['claims'])} claims")
    
    # Create a set of valid entity IDs
    valid_entity_ids = {entity['id'] for entity in graph_data['entities']}
    print(f"✅ Found {len(valid_entity_ids)} valid entity IDs")
    
    nodes = []
    links = []
    invalid_claims = 0
    
    # Add entities as nodes
    for entity in graph_data['entities']:
        # Get color based on type
        color = '#1f77b4'  # default blue
        if entity['type'] == 'person':
            color = '#ff7f0e'  # orange
        elif entity['type'] == 'team':
            color = '#2ca02c'  # green
        elif entity['type'] == 'project':
            color = '#d62728'  # red
        elif entity['type'] == 'topic':
            color = '#9467bd'  # purple
        elif entity['type'] == 'company':
            color = '#8c564b'  # brown
        elif entity['type'] == 'business_activity':
            color = '#e377c2'  # pink
        elif entity['type'] == 'identifier':
            color = '#7f7f7f'  # gray
        
        nodes.append({
            'id': entity['id'],
            'label': entity['names'][0] if entity['names'] else 'Unknown',
            'type': entity['type'],
            'confidence': entity.get('confidence', 0.5),
            'color': color,
            'size': max(8, min(20, 8 + (entity.get('confidence', 0.5) * 10)))
        })
    
    # Add claims as edges with validation
    for claim in graph_data['claims']:
        subject_id = claim.get('subject_id')
        object_id = claim.get('object_id')
        
        # Skip claims with missing entities
        if not subject_id or not object_id:
            invalid_claims += 1
            continue
            
        # Skip claims that reference non-existent entities
        if subject_id not in valid_entity_ids or object_id not in valid_entity_ids:
            invalid_claims += 1
            continue
        
        links.append({
            'source': subject_id,
            'target': object_id,
            'type': claim.get('type', 'relationship'),
            'predicate': claim.get('predicate', ''),
            'confidence': claim.get('confidence', 0.5),
            'id': claim['id'],
            'value': claim.get('value', '')
        })
    
    print(f"✅ Created {len(nodes)} nodes")
    print(f"✅ Created {len(links)} valid links (skipped {invalid_claims} invalid claims)")
    
    response_data = {
        'nodes': nodes, 
        'links': links,
        'stats': {
            'total_entities': len(graph_data['entities']),
            'total_claims': len(graph_data['claims']),
            'valid_links': len(links),
            'invalid_claims': invalid_claims,
            'displayed_entities': len(nodes)
        }
    }
    
    return jsonify(response_data)

@app.route('/api/claim/<claim_id>')
def get_claim(claim_id):
    """Get claim details with evidence"""
    # Find claim in graph_data
    claim = next((c for c in graph_data['claims'] if c['id'] == claim_id), None)
    if not claim:
        return jsonify({'error': 'Claim not found'}), 404
    
    # Find evidence
    evidence_list = []
    for evidence_id in claim.get('evidence_ids', []):
        evidence = next((e for e in graph_data['evidence'] if e['id'] == evidence_id), None)
        if evidence:
            evidence_list.append(evidence)
    
    # Find entities
    subject = next((e for e in graph_data['entities'] if e['id'] == claim.get('subject_id')), None)
    object_entity = next((e for e in graph_data['entities'] if e['id'] == claim.get('object_id')), None)
    
    return jsonify({
        'id': claim['id'],
        'type': claim.get('type', 'unknown'),
        'subject': subject['names'][0] if subject else 'Unknown',
        'predicate': claim.get('predicate', ''),
        'object': object_entity['names'][0] if object_entity else claim.get('value', ''),
        'valid_from': claim.get('valid_from'),
        'valid_to': claim.get('valid_to'),
        'confidence': claim.get('confidence', 0.5),
        'evidence': evidence_list
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint"""
    data = request.json
    query = data.get('query', '').lower()
    
    results = []
    
    # Search entities
    for entity in graph_data['entities']:
        if any(query in name.lower() for name in entity.get('names', [])):
            results.append({
                'id': entity['id'],
                'label': entity['names'][0] if entity['names'] else 'Unknown',
                'type': 'entity',
                'entity_type': entity.get('type', 'unknown'),
                'confidence': entity.get('confidence', 0.5)
            })
    
    # Search claims
    for claim in graph_data['claims']:
        # Find subject and object names for better search
        subject = next((e for e in graph_data['entities'] if e['id'] == claim.get('subject_id')), None)
        object_entity = next((e for e in graph_data['entities'] if e['id'] == claim.get('object_id')), None)
        
        search_text = f"{claim.get('predicate', '')} {claim.get('value', '')}"
        if subject:
            search_text += f" {subject['names'][0] if subject['names'] else ''}"
        if object_entity:
            search_text += f" {object_entity['names'][0] if object_entity['names'] else ''}"
        
        if query in search_text.lower():
            label = f"{subject['names'][0] if subject else 'Unknown'} {claim.get('predicate', '')} {object_entity['names'][0] if object_entity else claim.get('value', '')}"
            results.append({
                'id': claim['id'],
                'label': label[:50] + '...' if len(label) > 50 else label,
                'type': 'claim',
                'confidence': claim.get('confidence', 0.5)
            })
    
    # Limit results
    results = results[:20]
    
    return jsonify({'results': results})

@app.route('/api/entity/<entity_id>')
def get_entity(entity_id):
    """Get entity details"""
    entity = next((e for e in graph_data['entities'] if e['id'] == entity_id), None)
    if not entity:
        return jsonify({'error': 'Entity not found'}), 404
    
    # Find claims involving this entity
    claims_as_subject = [c for c in graph_data['claims'] if c.get('subject_id') == entity_id]
    claims_as_object = [c for c in graph_data['claims'] if c.get('object_id') == entity_id]
    
    # Get evidence count
    evidence_count = entity.get('evidence_count', 0)
    
    return jsonify({
        'id': entity['id'],
        'type': entity.get('type', 'unknown'),
        'names': entity.get('names', []),
        'confidence': entity.get('confidence', 0.5),
        'claims_as_subject': len(claims_as_subject),
        'claims_as_object': len(claims_as_object),
        'evidence_count': evidence_count
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)