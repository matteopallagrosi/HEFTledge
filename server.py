from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor

import json
from optimizer import OptimizerParams
from heftless import HEFTless

executor = ProcessPoolExecutor(max_workers=10)

def compute_task_assignment(path: str, post_data: str):
    params = OptimizerParams.from_json(post_data)
    if path == "/heftless":
        return HEFTless().run(params)
    raise ValueError("Unknown path: " + path)

class RequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        try:
            result = executor.submit(compute_task_assignment, path, post_data).result()
            response = json.dumps(result).encode('utf-8')
            self.send_response(200)
        except Exception as e:
            response = json.dumps({"error": str(e)}).encode('utf-8')
            self.send_response(400)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

# === Start server ===
def run(port=8080):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, RequestHandler)
    print(f"🚀 Server running on http://localhost:{port}/")
    httpd.serve_forever()

if __name__ == "__main__":
    run()