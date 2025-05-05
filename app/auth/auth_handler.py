import hashlib
import sqlite3
from typing import Optional, Tuple
import json
import os
from pathlib import Path

class AuthHandler:
    def __init__(self, db_path: str):
        # Get the base directory (salesvizz folder)
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Set database path
        self.db_path = os.path.join(data_dir, "salesvizz.db")
    
    def _get_db_connection(self):
        return sqlite3.connect(self.db_path)
    
    def register_user(self, username: str, password: str, role: str = 'viewer') -> bool:
        """Register a new user with hashed password"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed_password, role)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Tuple[int, str]]:
        """Authenticate user and return user_id and role if successful"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, role FROM users WHERE username = ? AND password = ?",
            (username, hashed_password)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result if result else None
    
    def get_user_permissions(self, role: str) -> list:
        """Get permissions for a given role"""
        from ..config import ROLES
        return ROLES.get(role, [])
    
    def check_permission(self, user_id: int, required_permission: str) -> bool:
        """Check if user has required permission"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
            
        role = result[0]
        permissions = self.get_user_permissions(role)
        return 'all' in permissions or required_permission in permissions 