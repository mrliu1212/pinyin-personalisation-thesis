import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from corpus.acquire import build_api_url
from corpus.cleaning import clean_wikisource_html, count_chinese_characters
from corpus.common import (
    DateMetadata,
    WorkMetadata,
    chronology_sort_key,
    date_needs_review,
    load_catalog,
)
from corpus.diagnostics import diagnostic_lines
from corpus.prepare import prepare


def catalog_payload(works: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "author_id": "zhu_ziqing",
        "author_name": "朱自清",
        "source_name": "Chinese Wikisource",
        "source_api": "https://zh.wikisource.org/w/api.php",
        "source_author_page": "https://zh.wikisource.org/wiki/Author:朱自清",
        "works": works,
    }


def work_payload(
    work_id: str,
    *,
    included: bool = True,
    chronology_value: str | None = "1925-10",
    chronology_precision: str = "month",
    chronology_certainty: str = "certain",
) -> dict[str, object]:
    return {
        "work_id": work_id,
        "work_title": "背影",
        "source_page_title": "背影",
        "source_page_url": "https://zh.wikisource.org/wiki/背影",
        "genre": "prose",
        "included": included,
        "exclusion_reason": None if included else "fixture exclusion",
        "chronology": {
            "value": chronology_value,
            "precision": chronology_precision,
            "certainty": chronology_certainty,
            "basis": "fixture",
        },
        "publication": {
            "value": "1925-11-22",
            "precision": "day",
            "certainty": "certain",
            "basis": "fixture",
        },
    }


class MetadataTest(unittest.TestCase):
    def test_catalog_metadata_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps(catalog_payload([work_payload("beiying")]), ensure_ascii=False),
                encoding="utf-8",
            )

            catalog = load_catalog(path)

        self.assertEqual(catalog.author_name, "朱自清")
        self.assertEqual(catalog.works[0].chronology.value, "1925-10")
        self.assertEqual(catalog.works[0].chronology.precision, "month")

    def test_missing_and_uncertain_dates_are_explicit_and_sort_last(self) -> None:
        certain = self.make_work("certain", DateMetadata("1925", "year", "certain", "x"))
        uncertain = self.make_work(
            "uncertain", DateMetadata("1924", "year", "uncertain", "x")
        )
        missing = self.make_work(
            "missing", DateMetadata(None, "unknown", "unknown", "x")
        )

        ordered = sorted([missing, uncertain, certain], key=chronology_sort_key)

        self.assertEqual([work.work_id for work in ordered], ["certain", "uncertain", "missing"])
        self.assertFalse(date_needs_review(certain.chronology))
        self.assertTrue(date_needs_review(uncertain.chronology))
        self.assertTrue(date_needs_review(missing.chronology))

    @staticmethod
    def make_work(work_id: str, chronology: DateMetadata) -> WorkMetadata:
        return WorkMetadata(
            author_id="zhu_ziqing",
            author_name="朱自清",
            work_id=work_id,
            work_title=work_id,
            source_page_title=work_id,
            source_page_url="https://example.test",
            genre="prose",
            included=True,
            exclusion_reason=None,
            chronology=chronology,
            publication=DateMetadata(None, "unknown", "unknown", "fixture"),
        )


class CleaningTest(unittest.TestCase):
    SAMPLE_HTML = """
        <div class="mw-parser-output">
          <div>更多資料：不是正文</div>
          <div id="headerContainer"><table><tr><td>作者和日期</td></tr></table></div>
          <p>第一段，保留原文。</p>
          <p>第二段有罕見詞：蓊蓊鬱鬱。</p>
          <div class="licenseContainer"><p>公有領域和頁面資訊</p></div>
          <div class="licensetpl">Public domain false false</div>
        </div>
    """

    def test_cleaning_is_deterministic_and_conservative(self) -> None:
        first = clean_wikisource_html(self.SAMPLE_HTML)
        second = clean_wikisource_html(self.SAMPLE_HTML)

        self.assertEqual(first, second)
        self.assertEqual(first, "第一段，保留原文。\n\n第二段有罕見詞：蓊蓊鬱鬱。\n")
        self.assertNotIn("作者和日期", first)
        self.assertNotIn("更多資料", first)
        self.assertNotIn("公有領域", first)
        self.assertNotIn("Public domain", first)
        self.assertEqual(count_chinese_characters(first), 18)

    def test_preparation_does_not_modify_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.json"
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            raw_dir.mkdir()
            catalog_path.write_text(
                json.dumps(catalog_payload([work_payload("beiying")]), ensure_ascii=False),
                encoding="utf-8",
            )
            payload = {
                "parse": {
                    "title": "背影",
                    "pageid": 1,
                    "revid": 2,
                    "text": self.SAMPLE_HTML,
                    "wikitext": "fixture source",
                }
            }
            raw_path = raw_dir / "beiying__rev_2.json"
            raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            raw_path.write_bytes(raw_bytes)
            acquisition = {
                "works": [
                    {
                        "work_id": "beiying",
                        "source_page_title": "背影",
                        "source_page_url": "https://zh.wikisource.org/wiki/背影",
                        "request_url": "https://zh.wikisource.org/w/api.php?fixture",
                        "page_id": 1,
                        "revision_id": 2,
                        "retrieved_at": "2026-08-10T00:00:00+00:00",
                        "content_variant": "canonical",
                        "raw_file": raw_path.name,
                        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    }
                ]
            }
            (raw_dir / "acquisition_manifest.json").write_text(
                json.dumps(acquisition), encoding="utf-8"
            )

            manifest = prepare(catalog_path, raw_dir, processed_dir)

            self.assertEqual(raw_path.read_bytes(), raw_bytes)
            self.assertEqual(manifest["works"][0]["chinese_character_count"], 18)


class AcquisitionAndDiagnosticTest(unittest.TestCase):
    def test_api_url_identifies_page_and_requested_provenance_fields(self) -> None:
        url = build_api_url("https://zh.wikisource.org/w/api.php", "背影")

        self.assertIn("page=%E8%83%8C%E5%BD%B1", url)
        self.assertIn("prop=text%7Cwikitext%7Crevid", url)
        self.assertNotIn("variant=", url)

    def test_diagnostics_report_counts_chronology_and_exclusions(self) -> None:
        manifest = {
            "author_id": "zhu_ziqing",
            "author_name": "朱自清",
            "works": [
                {
                    "work_title": "匆匆",
                    "included": True,
                    "chinese_character_count": 10,
                    "chronology_needs_review": False,
                    "publication_needs_review": False,
                    "chronology": {
                        "value": "1922-03-28",
                        "precision": "day",
                        "certainty": "certain",
                    },
                },
                {
                    "work_title": "獨自",
                    "included": False,
                    "exclusion_reason": "poetry",
                    "chinese_character_count": 0,
                    "chronology_needs_review": False,
                    "publication_needs_review": False,
                    "chronology": {
                        "value": "1922-02-22",
                        "precision": "day",
                        "certainty": "certain",
                    },
                },
            ],
        }

        output = "\n".join(diagnostic_lines(manifest))

        self.assertIn("Included works: 1", output)
        self.assertIn("Excluded works: 1", output)
        self.assertIn("Total Chinese characters: 10", output)
        self.assertIn("匆匆 — 10 chars", output)
        self.assertIn("獨自: poetry", output)


if __name__ == "__main__":
    unittest.main()
