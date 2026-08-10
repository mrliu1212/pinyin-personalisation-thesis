"""Conservative extraction of author text from MediaWiki-rendered HTML."""

from __future__ import annotations

import re
from html.parser import HTMLParser


BLOCK_TAGS = {"blockquote", "br", "dd", "div", "dl", "dt", "h1", "h2", "h3", "h4", "hr", "li", "p", "pre"}
SKIP_TAGS = {"figure", "script", "style", "table"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
SKIP_CLASSES = {
    "catlinks",
    "licensebanner",
    "licensecontainer",
    "licensetpl",
    "mw-editsection",
    "noprint",
    "pagenum",
    "printfooter",
    "ws-footer",
    "ws-header",
    "ws-noexport",
}
SKIP_IDS = {"headerContainer", "mw-navigation"}
CHINESE_CHARACTER_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class WikisourceTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._body_started = False
        self._header_skip_active = False
        self.found_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = {item.lower() for item in (attributes.get("class") or "").split()}
        should_skip = (
            tag in SKIP_TAGS
            or attributes.get("id") in SKIP_IDS
            or bool(classes & SKIP_CLASSES)
        )
        if self._skip_depth:
            if tag not in VOID_TAGS:
                self._skip_depth += 1
            return
        if should_skip:
            if attributes.get("id") == "headerContainer":
                self.found_header = True
                self._header_skip_active = True
            if tag not in VOID_TAGS:
                self._skip_depth = 1
            return
        if self._body_started and tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            if self._skip_depth == 0 and self._header_skip_active:
                self._header_skip_active = False
                self._body_started = True
            return
        if self._body_started and tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._body_started and not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        text = "".join(self._parts).replace("\u200b", "").replace("\xa0", " ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        nonempty_runs: list[str] = []
        previous_blank = True
        for line in lines:
            if line:
                nonempty_runs.append(line)
                previous_blank = False
            elif not previous_blank:
                nonempty_runs.append("")
                previous_blank = True
        return "\n".join(nonempty_runs).strip() + "\n"


def clean_wikisource_html(html: str) -> str:
    extractor = WikisourceTextExtractor()
    extractor.feed(html)
    extractor.close()
    if not extractor.found_header:
        raise ValueError("Wikisource headerContainer not found; refusing unsafe extraction")
    return extractor.text()


def count_chinese_characters(text: str) -> int:
    return len(CHINESE_CHARACTER_RE.findall(text))
