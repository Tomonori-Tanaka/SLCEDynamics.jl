# References

Supporting literature for this package — papers, books, and useful web
resources.

## Layout

- `*.md` (this directory) — **one note per source**: citation, link, a short
  summary, and why it matters for this package. **Tracked in git** (publishable,
  no copyright issue — these are your own notes and metadata).
- `papers/` — the actual paper **PDFs** and other copyrighted files.
  **Gitignored**; local-only, never committed or published.

A fresh clone therefore carries the curated notes and links, but not the
copyrighted files. Drop PDFs into `papers/` locally; they stay on your machine.

## Index

> One line per source, linking to its note. Fill in as you add references.

### Papers
- [`<short-key>.md`](<short-key>.md) — <Author Year>, "<title>" — <one-line why>.

### Web resources / sites
- <name> — <url> — <what it's good for>.

## Adding a reference

1. Copy [`_paper-template.md`](_paper-template.md) to `<short-key>.md`
   (e.g. `tanaka2024-sce.md`) and fill it in.
2. Put the PDF at `papers/<short-key>.pdf` (stays local, gitignored).
3. Add a line to the index above.
