"""
Multi-File Project Refactoring Benchmark Challenge.
Simulates cross-module circular imports and signature hallucination in multi-file repositories.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MultiFileProject:
    name: str
    files: Dict[str, str]  # module_name -> source_code


# 1. Broken Multi-File Project (contains circular import and unresolved cross-module symbol)
BROKEN_LLM_MULTI_FILE_PROJECT = MultiFileProject(
    name="ecommerce_auth_microservice_broken",
    files={
        "auth_service": """
from database import get_user_record

def generate_auth_token(user_id, role):
    user = get_user_record(user_id)
    if not user:
        return None
    return f"token_{user_id}_{role}"

def verify_token(token):
    return token.startswith("token_")
""",
        "database": """
# LLM introduced circular import: database imports from auth_service!
from auth_service import verify_token

DB = {"user_1": {"name": "Alice", "role": "admin"}}

def get_user_record(user_id):
    return DB.get(user_id)

def secure_query(user_id, token):
    if not verify_token(token):
        return None
    return get_user_record(user_id)
""",
        "api_gateway": """
# LLM hallucinates non-existent function 'authenticate_user' in auth_service
from auth_service import authenticate_user

def route_request(path, token):
    if path == "/profile":
        return authenticate_user(token)
    return "404 Not Found"
"""
    }
)

# 2. Invariant-Repaired Multi-File Project (ICIA Gov)
REPAIRED_ICIA_MULTI_FILE_PROJECT = MultiFileProject(
    name="ecommerce_auth_microservice_repaired",
    files={
        "models": """
# Decoupled shared domain invariants (breaks circular import)
TOKEN_PREFIX = "token_"

def validate_token_format(token):
    return token is not None and token.startswith(TOKEN_PREFIX)
""",
        "auth_service": """
from models import validate_token_format, TOKEN_PREFIX
from database import get_user_record

def generate_auth_token(user_id, role):
    user = get_user_record(user_id)
    if not user:
        return None
    return f"{TOKEN_PREFIX}{user_id}_{role}"

def verify_token(token):
    return validate_token_format(token)

def authenticate_user(token):
    if verify_token(token):
        return {"status": "authenticated"}
    return None
""",
        "database": """
from models import validate_token_format

DB = {"user_1": {"name": "Alice", "role": "admin"}}

def get_user_record(user_id):
    return DB.get(user_id)

def secure_query(user_id, token):
    if not validate_token_format(token):
        return None
    return get_user_record(user_id)
""",
        "api_gateway": """
from auth_service import authenticate_user

def route_request(path, token):
    if path == "/profile":
        return authenticate_user(token)
    return "404 Not Found"
"""
    }
)
