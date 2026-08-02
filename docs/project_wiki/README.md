# Participation-Constrained Cell-Free ISAC Wiki and Poster Defense Manual

This directory contains the reproducible ElegantBook source for the project
wiki, poster-presentation script, and bilingual defense Q&A manual.

## Build

Use XeLaTeX because the manual contains Chinese defense prompts:

```powershell
cd docs/project_wiki
latexmk -xelatex -interaction=nonstopmode main.tex
```

The generated file is `main.pdf`.  To remove generated build files, run
`latexmk -C` from this directory.  The final reviewed PDF is copied to
`output/Participation_Constrained_CF_ISAC_Wiki_and_Defense_Manual.pdf`.

## Source hierarchy

- `main.tex` and `chapters/`: authored project manual.
- `vendor/ElegantBook/`: pinned upstream template snapshot.
- `output/`: reviewed distributable PDF.
- `../../deliverables/paper_results_and_figures_v1_0/`: the only numerical
  evidence source used by this manual.

## ElegantBook provenance

The template is an unmodified snapshot of
[`ElegantLaTeX/ElegantBook`](https://github.com/ElegantLaTeX/ElegantBook),
commit `8b90c11e4a5ffd9d1e07174011303c133093d09c` (retrieved 2026-07-30),
under LPPL-1.3c or later.  See `vendor/ElegantBook/License` and `UPSTREAM.md`.

## Scope

This manual explains the current project model and evidence; it does not claim
global optimality of the original MINLP, nor does it treat the SDR as a
deployable baseline.  The Git-tracked final result package must be present for
the local figure links to resolve.
