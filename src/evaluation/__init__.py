"""Evaluation and explainability modules."""

from src.evaluation.explainer import SHAPExplainer, explain_model

__all__ = ["explain_model", "SHAPExplainer"]
