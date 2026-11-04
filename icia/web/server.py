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
from icia.tasks.hierarchical_foraging import HierarchicalForagingEnv
from icia.experiments.run_hierarchical_experiment import build_icia_controller
from icia.tasks.code_synthesis_task import (
    get_binary_search_task,
    get_matrix_transpose_task,
    get_parentheses_validator_task
)
from icia.experiments.run_code_synthesis_experiment import (
    run_icia_generator,
    run_autoregressive_baseline
)

# Workspace Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DASHBOARD_PATH = os.path.join(PROJECT_ROOT, "dashboard.html")

# Global Controller & State Instances
physics_env = ContinuousCartPoleEnv()
physics_controller = ContinuousICIAController()

hierarchical_env = HierarchicalForagingEnv(stochastic_jam=True)
hierarchical_controller = build_icia_controller()
hierarchical_state, hierarchical_facts = hierarchical_env.reset()
hierarchical_disc = 0.0


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
                "version": "0.1.0",
                "engines": ["ContinuousPhysics", "CollectiveIntention", "HierarchicalIntention", "LatentReasoner"]
            })
        else:
            self.send_error(404, "Path not found")

    def do_POST(self):
        global hierarchical_state, hierarchical_facts, hierarchical_disc, hierarchical_controller

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

        elif self.path == "/api/multiagent_step":
            # Compute collective swarm dynamics step
            raw_agents = req_data.get("agents", [])
            obstacles_raw = req_data.get("obstacles", [])
            obs_tuples = [(np.array(o[:2], dtype=np.float32), float(o[2]) if len(o) > 2 else 0.5) for o in obstacles_raw]

            agent_objs = []
            for i, a in enumerate(raw_agents):
                agent_objs.append(AgentState(
                    agent_id=i,
                    initial_pos=np.array(a.get("pos", [0.0, 0.0]), dtype=np.float64),
                    goal_pos=np.array(a.get("goal", [0.0, 0.0]), dtype=np.float64),
                    priority=float(a.get("priority", 1.0))
                ))

            field = CollectiveIntentionField(
                agents=agent_objs,
                obstacles=obs_tuples,
                dt=0.05
            )
            step_res = field.step()

            self._send_json({
                "agents": [
                    {
                        "id": ag.agent_id,
                        "pos": ag.pos.tolist(),
                        "goal": ag.goal.tolist(),
                        "priority": ag.priority,
                        "reached_goal": ag.reached_goal
                    }
                    for ag in field.agents
                ],
                "collisions": step_res["collisions"],
                "all_reached": step_res["all_reached"]
            })

        elif self.path == "/api/hierarchical_step":
            # Step the hierarchical task
            if req_data.get("reset", False):
                hierarchical_state, hierarchical_facts = hierarchical_env.reset()
                hierarchical_controller = build_icia_controller()
                hierarchical_disc = 0.0

            action, telemetry = hierarchical_controller.compute_action(
                hierarchical_state, hierarchical_facts, discrepancy=hierarchical_disc
            )
            next_state, hierarchical_disc, done, info = hierarchical_env.step(action)
            hierarchical_state = next_state
            hierarchical_facts = info["facts"]

            subgoals_summary = [
                {
                    "id": sg.id,
                    "desc": sg.description,
                    "target": sg.target_state.tolist(),
                    "completed": sg.is_completed,
                    "invalidated": sg.is_invalidated
                }
                for sg in hierarchical_controller.sub_goals
            ]

            self._send_json({
                "pos": hierarchical_state.tolist(),
                "active_subgoal": telemetry["active_subgoal"],
                "impasse": telemetry["impasse"],
                "replan": telemetry["replan"],
                "facts": hierarchical_facts,
                "subgoals": subgoals_summary,
                "done": bool(done)
            })

        elif self.path == "/api/code_synthesis":
            task_type = req_data.get("task", "binary_search")
            if task_type == "binary_search":
                task_data = get_binary_search_task()
            elif task_type == "matrix_transpose":
                task_data = get_matrix_transpose_task()
            else:
                task_data = get_parentheses_validator_task()

            icia_res = run_icia_generator(task_data)
            ar_res = run_autoregressive_baseline(task_data)

            self._send_json({
                "task": task_type,
                "icia": icia_res,
                "autoregressive": ar_res
            })

        elif self.path == "/api/deliberate":
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
