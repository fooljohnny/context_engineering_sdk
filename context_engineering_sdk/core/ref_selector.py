"""Reference selector parsing and content extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from context_engineering_sdk.core.errors import SelectorResolveError


@dataclass
class ParsedSelector:
    type: str  # "lines" | "chars" | "json" | "regex"
    params: dict


class RefSelector:
    """Parses selector strings and extracts content fragments."""

    def parse(self, selector: str) -> list[ParsedSelector]:
        parts = [s.strip() for s in selector.split(",")]
        result: list[ParsedSelector] = []
        for part in parts:
            if not part:
                continue
            if ":" not in part:
                raise SelectorResolveError(f"Invalid selector format: {part!r}")
            type_str, _, value = part.partition(":")
            type_str = type_str.strip()
            value = value.strip()
            if type_str == "lines":
                m = re.match(r"(\d+)-(\d+)", value)
                if not m:
                    raise SelectorResolveError(
                        f"Invalid lines selector: {value!r}"
                    )
                result.append(
                    ParsedSelector(
                        type="lines",
                        params={"from": int(m.group(1)), "to": int(m.group(2))},
                    )
                )
            elif type_str == "chars":
                m = re.match(r"(\d+)-(\d+)", value)
                if not m:
                    raise SelectorResolveError(
                        f"Invalid chars selector: {value!r}"
                    )
                result.append(
                    ParsedSelector(
                        type="chars",
                        params={"from": int(m.group(1)), "to": int(m.group(2))},
                    )
                )
            elif type_str == "json":
                result.append(
                    ParsedSelector(type="json", params={"path": value})
                )
            elif type_str == "regex":
                result.append(
                    ParsedSelector(type="regex", params={"pattern": value})
                )
            else:
                raise SelectorResolveError(
                    f"Unknown selector type: {type_str!r}"
                )
        return result

    def extract(self, content: str, selector: str) -> str:
        parsed = self.parse(selector)
        if not parsed:
            return content

        result = content
        for sel in parsed:
            if sel.type == "lines":
                lines = result.splitlines()
                start = max(0, sel.params["from"] - 1)  # 1-indexed
                end = min(len(lines), sel.params["to"])
                result = "\n".join(lines[start:end])
            elif sel.type == "chars":
                start = sel.params["from"]
                end = sel.params["to"]
                result = result[start:end]
            elif sel.type == "regex":
                pattern = sel.params["pattern"]
                try:
                    m = re.search(pattern, result)
                except re.error as e:
                    raise SelectorResolveError(
                        f"Invalid regex pattern: {e}"
                    ) from e
                if m:
                    result = m.group(0)
                else:
                    result = ""
            elif sel.type == "json":
                result = self._extract_json_path(result, sel.params["path"])
        return result

    @staticmethod
    def _extract_json_path(content: str, path: str) -> str:
        """Minimal JSONPath extraction supporting $.key.key[idx] patterns."""
        import json

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            raise SelectorResolveError(
                f"Content is not valid JSON: {e}"
            ) from e

        if not path.startswith("$."):
            raise SelectorResolveError(
                f"JSONPath must start with '$.' : {path!r}"
            )

        tokens = re.findall(r'\.(\w+)|\[(\d+)\]', path[1:])
        current = data
        for key_match, idx_match in tokens:
            try:
                if key_match:
                    current = current[key_match]
                else:
                    current = current[int(idx_match)]
            except (KeyError, IndexError, TypeError) as e:
                raise SelectorResolveError(
                    f"JSONPath resolve failed at {path!r}: {e}"
                ) from e
        if isinstance(current, str):
            return current
        return json.dumps(current, ensure_ascii=False)
