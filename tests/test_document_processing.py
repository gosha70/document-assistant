"""Tests for document loading, splitting, and file type handling."""
import os
import logging
import pytest
from unittest.mock import MagicMock, patch

from embeddings.unstructured.file_type import FileType
from embeddings.unstructured.base_file_converter import BaseFileConverter
from embeddings.unstructured.document_splitter import DocumentSplitter
from src.config.settings import get_settings


class TestFileType:
    def test_all_16_types_defined(self):
        expected = {"csv", "ddl", "xlsx", "java", "js", "json", "html", "md", "pdf", "py", "rtf", "sql", "txt", "xml", "xsl", "yaml"}
        actual = {ft.value for ft in FileType}
        assert actual == expected

    def test_get_file_type_valid(self):
        assert FileType.get_file_type("pdf") == FileType.PDF
        assert FileType.get_file_type("java") == FileType.JAVA

    def test_get_file_type_invalid(self):
        assert FileType.get_file_type("docx") is None

    def test_get_extension(self):
        assert FileType.PDF.get_extension() == ".pdf"
        assert FileType.JAVA.get_extension() == ".java"

    def test_get_file_type_by_extension(self):
        assert FileType.get_file_type_by_extension("report.pdf") == FileType.PDF
        assert FileType.get_file_type_by_extension("code.py") == FileType.PYTHON
        assert FileType.get_file_type_by_extension("unknown.xyz") is None


class TestBaseFileConverter:
    def test_get_text_splitter_returns_splitter(self):
        splitter = BaseFileConverter.get_text_splitter(FileType.TEXT)
        assert splitter is not None
        settings = get_settings()
        assert splitter._chunk_size == settings.chunking.chunk_size
        assert splitter._chunk_overlap == settings.chunking.chunk_overlap

    def test_get_text_splitter_with_language(self):
        splitter = BaseFileConverter.get_text_splitter(FileType.PYTHON)
        assert splitter is not None

    def test_get_text_splitter_java(self):
        splitter = BaseFileConverter.get_text_splitter(FileType.JAVA)
        assert splitter is not None


class TestDocumentSplitter:
    @pytest.fixture
    def splitter(self):
        return DocumentSplitter(logging)

    def test_all_converters_registered(self, splitter):
        for ft in FileType:
            converter = splitter.get_converter(ft)
            assert converter is not None, f"No converter for {ft}"

    def test_process_text_file(self, splitter, sample_text_file):
        docs = splitter.process_file(sample_text_file)
        assert docs is not None
        assert len(docs) > 0
        assert all(hasattr(d, "page_content") for d in docs)

    def test_process_markdown_file(self, splitter, sample_markdown_file):
        docs = splitter.process_file(sample_markdown_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_csv_file(self, splitter, sample_csv_file):
        docs = splitter.process_file(sample_csv_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_html_file(self, splitter, sample_html_file):
        docs = splitter.process_file(sample_html_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_python_file(self, splitter, sample_python_file):
        docs = splitter.process_file(sample_python_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_java_file(self, splitter, sample_java_file):
        docs = splitter.process_file(sample_java_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_json_file(self, splitter, sample_json_file):
        docs = splitter.process_file(sample_json_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_xml_file(self, splitter, sample_xml_file):
        docs = splitter.process_file(sample_xml_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_sql_file(self, splitter, sample_sql_file):
        docs = splitter.process_file(sample_sql_file)
        assert docs is not None
        assert len(docs) > 0

    def test_process_js_file(self, splitter, sample_js_file):
        docs = splitter.process_file(sample_js_file)
        assert docs is not None
        assert len(docs) > 0

    def test_find_files(self, splitter, tmp_dir, sample_text_file):
        files = splitter.find_files(tmp_dir, ".txt")
        assert len(files) == 1
        assert files[0].endswith(".txt")


class TestConverterFallbackPath:
    """Verify that converters using log_info(exc_info=False) don't raise TypeError.

    Bug fix: base_file_converter.log_info() did not accept **kwargs, so the
    exc_info=False kwarg passed by CSV/HTML/JSON/XML/Markdown converters caused
    a TypeError when the primary loader failed and the fallback path triggered.
    """

    @pytest.fixture
    def splitter(self):
        return DocumentSplitter(logging)

    def _force_fallback(self, converter_cls, file_type, tmp_dir):
        """Write a file whose content will cause the primary Unstructured loader
        to fail, forcing the converter through its except branch (which calls
        log_info with exc_info=False) and into the TextLoader fallback."""
        from unittest.mock import patch
        from embeddings.unstructured.base_file_converter import BaseFileConverter

        ext = file_type.get_extension()
        path = os.path.join(tmp_dir, f"fallback_test{ext}")
        # Write valid text so TextLoader fallback succeeds
        with open(path, "w") as f:
            f.write("fallback test content")

        converter = converter_cls(logging=logging)
        text_splitter = BaseFileConverter.get_text_splitter(file_type)

        # Patch the primary loader to force the except branch
        primary_loader_mod = {
            "csv": "embeddings.unstructured.csv_converter.UnstructuredCSVLoader",
            "html": "embeddings.unstructured.html_converter.UnstructuredHTMLLoader",
            "json": "embeddings.unstructured.json_converter.JSONLoader",
            "xml": "embeddings.unstructured.xml_converter.UnstructuredXMLLoader",
            "md": "embeddings.unstructured.markdown_converter.UnstructuredMarkdownLoader",
        }
        loader_path = primary_loader_mod[file_type.value]
        mock_loader = MagicMock()
        mock_loader.return_value.load_and_split.side_effect = RuntimeError("forced failure")

        with patch(loader_path, mock_loader):
            # This must NOT raise TypeError from log_info(**kwargs)
            docs = converter.load_and_split_file(text_splitter=text_splitter, file_path=path)

        assert docs is not None
        assert len(docs) > 0

    def test_csv_fallback_no_typeerror(self, tmp_dir):
        from embeddings.unstructured.csv_converter import CsvConverter
        self._force_fallback(CsvConverter, FileType.CSV, tmp_dir)

    def test_html_fallback_no_typeerror(self, tmp_dir):
        from embeddings.unstructured.html_converter import HtmlConverter
        self._force_fallback(HtmlConverter, FileType.HTML, tmp_dir)

    def test_json_fallback_no_typeerror(self, tmp_dir):
        from embeddings.unstructured.json_converter import JsonConverter
        self._force_fallback(JsonConverter, FileType.JSON, tmp_dir)

    def test_xml_fallback_no_typeerror(self, tmp_dir):
        from embeddings.unstructured.xml_converter import XmlConverter
        self._force_fallback(XmlConverter, FileType.XML, tmp_dir)

    def test_markdown_fallback_no_typeerror(self, tmp_dir):
        from embeddings.unstructured.markdown_converter import MarkdownConverter
        self._force_fallback(MarkdownConverter, FileType.MARKDOWN, tmp_dir)
