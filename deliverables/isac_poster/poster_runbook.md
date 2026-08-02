# Poster Runbook

## Produced locally without external API calls

`isac_research_poster.pptx` was composed directly from the audited paper PDF
and local experiment figures.  It is editable in PowerPoint.

## Paper2Poster-compatible input layout

To run the upstream multi-agent pipeline later, place the paper at:

```text
Paper2Poster-data/isac_poster/paper.pdf
Paper2Poster-data/isac_poster/poster.yaml
```

The upstream command would be:

```bash
python -m PosterAgent.new_pipeline \
  --poster_path="Paper2Poster-data/isac_poster/paper.pdf" \
  --model_name_t="4o" \
  --model_name_v="4o" \
  --poster_width_inches=48 \
  --poster_height_inches=36
```

This command is intentionally **not executed**: the environment has no
`OPENAI_API_KEY`, and this poster instead uses the grounded local workflow.

## Final QA before print

1. Insert author, affiliation, institutional logo, and QR code.
2. Verify all figure labels remain readable at the intended print size.
3. Confirm conference size, bleed, and color-profile requirements.
4. Retain the conditional-mean and feasibility reporting convention in the footer.
