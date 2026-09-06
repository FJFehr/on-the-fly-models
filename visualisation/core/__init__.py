"""Shared rendering used by both live training and the paper's figures.

arc.py/style.py back visualisation/__init__.py's public API (imported by
models/ and training/logging.py during every run); embedding_clusters.py
backs both training's end-of-run cluster diagnostic and
visualisation/paper/plot_embedding_clusters.py's paired comparison figures.
Nothing here is paper-specific -- see visualisation/paper/ for that.
"""
