from pathlib import Path
import unittest

from app.runtime_config import validate_required_env


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class SecurityHardeningTests(unittest.TestCase):
    def test_dockerignore_protects_local_secrets_and_data(self):
        dockerignore = read_text(".dockerignore")

        self.assertIn(".env", dockerignore)
        self.assertIn(".conda/", dockerignore)
        self.assertIn(".git", dockerignore)
        self.assertIn("data/", dockerignore)

    def test_runtime_config_requires_real_secrets(self):
        with self.assertRaises(RuntimeError) as exc:
            validate_required_env({})

        message = str(exc.exception)
        self.assertIn("OPENAI_API_KEY", message)
        self.assertIn("DATABASE_URL", message)
        self.assertIn("JWT_SECRET_KEY", message)

        validate_required_env(
            {
                "OPENAI_API_KEY": "test-openai-key",
                "DATABASE_URL": "postgresql://user:pass@example/db",
                "JWT_SECRET_KEY": "x" * 32,
            }
        )

    def test_auth_has_no_default_jwt_secret(self):
        auth_py = read_text("app/auth.py")

        self.assertNotIn("CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING", auth_py)
        self.assertIn('os.getenv("JWT_SECRET_KEY", "").strip()', auth_py)

    def test_roles_and_rag_allowlist_are_aligned(self):
        model_py = read_text("models/database.py")
        rag_py = read_text("app/secure_rag.py")

        for role in ("employee", "manager", "hr", "security", "finance", "executive", "admin"):
            self.assertTrue(f'{role}"' in model_py or f"{role}'" in model_py)
            self.assertIn(f'"{role}"', rag_py)

        self.assertIn("ROLE_SOURCE_ALLOWLIST", rag_py)
        self.assertIn('"security": ["security_policy.txt", "engineering_standards.docx"]', rag_py)

    def test_sensitive_demo_sources_are_excluded_from_ingestion_and_retrieval(self):
        ingestion_py = read_text("app/ingestion.py")
        rag_py = read_text("app/secure_rag.py")

        for filename in ("claude_api_tokens_2026_04.csv", "Credits"):
            self.assertIn(filename, ingestion_py)
            self.assertIn(filename, rag_py)

        self.assertIn("should_skip_source", ingestion_py)
        self.assertIn("EXCLUDED_SOURCE_FILENAMES", rag_py)

    def test_health_endpoint_does_not_call_external_ai_services(self):
        server_py = read_text("app/server.py")

        self.assertNotIn('embed_query("health check")', server_py)
        self.assertNotIn('_llm.invoke("ping")', server_py)
        self.assertNotIn('similarity_search("test"', server_py)
        self.assertIn('@app.get("/ready"', server_py)

    def test_readme_documents_jwt_and_sse_contract(self):
        readme = read_text("README.md")
        request_section = readme.split("Request:", 1)[1].split("Response", 1)[0]

        self.assertNotIn('"role": "employee"', request_section)
        self.assertIn("The role comes from the JWT", readme)
        self.assertIn("Server-Sent Events", readme)


if __name__ == "__main__":
    unittest.main()
