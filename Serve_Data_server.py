import http.server
import socketserver
import os


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', 'https://127.0.0.1:5000')  # Allow requests from any origin
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')  # Allow specific HTTP methods
        super().end_headers()


def run_server(port=8080):
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)
    handler = CORSHTTPRequestHandler

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Server running on http://localhost:{port}/")
        httpd.serve_forever()
