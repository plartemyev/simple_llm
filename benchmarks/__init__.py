"""Benchmark infrastructure for local LLM serving.

Modules:
  slicer   — turns public caption text into benchmark inputs:
             whole-phrase samples and dictation-style fragment streams.
  runner   — latency/throughput scenarios against a running server.
  quality  — replays message lists against a server and scores outputs
             against a gold set (token overlap + output-format rules).
  report   — aggregates runner/quality JSON results into markdown.
"""
