import http.server
import socketserver
from pathlib import Path

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Serve files from project root
        root = Path(__file__).parent.resolve()
        return str(root / path.lstrip("/"))

if __name__ == "__main__":
    print(f"Serving on http://127.0.0.1:{PORT}/viewer/")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
