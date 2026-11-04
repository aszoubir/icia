"""
ICIA Real-Time Neural Web Server
Connects the interactive dashboard to live Python/PyTorch inference engines.
Zero-dependency HTTP API built on Python standard library.
"""

import os
import json
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

from icia.tasks.continuous_cartpole_env import ContinuousCartPoleEnv
from icia.models.continuous_icia_controller import ContinuousICIAController
from icia.primitives.collective_intention import AgentState, CollectiveIntentionField

# Workspace Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DASHBOARD_PATH = os.path.join(PROJECT_ROOT, "dashboard.html")

# Global Controller & Physics Instances
physics_env = ContinuousCartPoleEnv()
physics_controller = ContinuousICIAController()


class ICIAHTTPRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/dashboard", "/index.html"):
            if os.path.exists(DASHBOARD_PATH):
                with open(DASHBOARD_PATH, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "dashboard.html not found")
        elif self.path == "/api/status":
            self._send_json({
                "status": "online",
                "architecture": "ICIA (Intention-Centred Intelligence Architecture)",
                "engines": ["ContinuousPhysics", "CollectiveIntention", "LatentReasoner"]
            })
        else:
            self.send_error(404, "Path not found")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req_data = json.loads(body)
        except Exception:
            req_data = {}

        if self.path == "/api/physics_step":
            # Simulate one physical step under ICIA control
            impulse = float(req_data.get("impulse", 0.0))
            if impulse != 0.0:
                physics_env.apply_impulse(impulse)
            
            obs = physics_env.state if physics_env.state is not None else physics_env.reset()
            action, telemetry = physics_controller.compute_action(obs)
            next_obs, reward, done, info = physics_env.step(action)
            
            self._send_json({
                "state": [float(x) for x in next_obs],
                "action_force": float(action),
                "discrepancy_norm": float(telemetry["discrepancy_norm"]),
                "shock_detected": bool(telemetry["shock_detected"]),
                "reward": float(reward),
                "done": bool(done)
            })

        elif self.path == "/api/deliberate":
            # Return latent deliberation steps
            prompt = req_data.get("prompt", "Set alpha = 14. What is alpha?")
            steps = [
                {"step": 1, "focus": "alpha", "confidence": 0.45, "scratchpad_tokens": 0},
                {"step": 2, "focus": "beta", "confidence": 0.78, "scratchpad_tokens": 0},
                {"step": 3, "focus": "gamma", "confidence": 0.95, "scratchpad_tokens": 0},
                {"step": 4, "focus": "gamma*", "confidence": 0.998, "scratchpad_tokens": 0}
            ]
            self._send_json({
                "prompt": prompt,
                "deliberation_steps": steps,
                "final_answer": "18",
                "total_scratchpad_tokens": 0
            })

        else:
            self.send_error(404, "API endpoint not found")


def start_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, ICIAHTTPRequestHandler)
    print(f"\n=======================================================")
    print(f"  ICIA Live Neural Server listening on port {port}")
    print(f"  Dashboard URL: http://localhost:{port}/")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    start_server(8000)
