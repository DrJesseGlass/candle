#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# HuggingFace cache location
HOME = Path.home()

# Find the correct snapshot directory
GGUF_BASE = HOME / '.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF'

# Find the snapshot directory (there should be one)
snapshots = list((GGUF_BASE / 'snapshots').glob('*'))
if not snapshots:
    print(f"Error: No snapshots found in {GGUF_BASE / 'snapshots'}", file=sys.stderr)
    sys.exit(1)

SNAPSHOT_PATH = snapshots[0]
print(f"Model location: {SNAPSHOT_PATH}")
# Verify files exist
for file in ['model.safetensors', 'tokenizer.json', 'config.json']:
    if not (SNAPSHOT_PATH / file).exists():
        print(f"Error: {file} not found at {SNAPSHOT_PATH}", file=sys.stderr)
        sys.exit(1)

class CustomHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        SimpleHTTPRequestHandler.end_headers(self)
    
    def do_GET(self):
        # Serve model files from HuggingFace cache
        if self.path == '/model.safetensors':
            self.send_file(SNAPSHOT_PATH / 'model.safetensors')
        elif self.path == '/tokenizer.json':
            self.send_file(SNAPSHOT_PATH / 'tokenizer.json')
        elif self.path == '/config.json':
            self.send_file(SNAPSHOT_PATH / 'config.json')
        else:
            # Serve everything else from current directory (pkg/, index.html)
            SimpleHTTPRequestHandler.do_GET(self)
    
    def send_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            if filepath.suffix == '.json':
                self.send_header('Content-Type', 'application/json')
            else:
                self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(404, f"File not found: {e}")

if __name__ == '__main__':
    PORT = 8080
    print(f"Serving WASM from: {os.getcwd()}")
    print(f"Serving models from: {SNAPSHOT_PATH}")
    print(f"Server running at http://localhost:{PORT}")
    
    server = HTTPServer(('', PORT), CustomHandler)
    server.serve_forever()
