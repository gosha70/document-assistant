import os
import sys
import tempfile
import pytest

# Add project root to path so imports work without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_text_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.txt")
    with open(path, "w") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 50)
    return path


@pytest.fixture
def sample_pdf_file(tmp_dir):
    """Create a minimal valid PDF for testing."""
    path = os.path.join(tmp_dir, "sample.pdf")
    # Minimal valid PDF
    content = b"""%PDF-1.0
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""
    with open(path, "wb") as f:
        f.write(content)
    return path


@pytest.fixture
def sample_json_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.json")
    with open(path, "w") as f:
        f.write('{"name": "test", "description": "A test JSON document with enough content to split."}')
    return path


@pytest.fixture
def sample_markdown_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.md")
    with open(path, "w") as f:
        f.write("# Test Document\n\nThis is a test markdown document.\n\n## Section 1\n\nContent here.\n")
    return path


@pytest.fixture
def sample_csv_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.csv")
    with open(path, "w") as f:
        f.write("name,age,city\nAlice,30,New York\nBob,25,San Francisco\n")
    return path


@pytest.fixture
def sample_html_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.html")
    with open(path, "w") as f:
        f.write("<html><body><h1>Test</h1><p>This is test content.</p></body></html>")
    return path


@pytest.fixture
def sample_python_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.py")
    with open(path, "w") as f:
        f.write('def hello():\n    """Say hello."""\n    return "Hello, World!"\n')
    return path


@pytest.fixture
def sample_java_file(tmp_dir):
    path = os.path.join(tmp_dir, "Sample.java")
    with open(path, "w") as f:
        f.write('public class Sample {\n    public static void main(String[] args) {\n        System.out.println("Hello");\n    }\n}\n')
    return path


@pytest.fixture
def sample_xml_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.xml")
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<root><item>Test content</item></root>\n')
    return path


@pytest.fixture
def sample_sql_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.sql")
    with open(path, "w") as f:
        f.write("SELECT * FROM users WHERE id = 1;\nINSERT INTO users (name) VALUES ('test');\n")
    return path


@pytest.fixture
def sample_js_file(tmp_dir):
    path = os.path.join(tmp_dir, "sample.js")
    with open(path, "w") as f:
        f.write('function hello() {\n  console.log("Hello, World!");\n}\n')
    return path
