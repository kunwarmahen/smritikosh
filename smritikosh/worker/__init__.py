"""Standalone background-job worker for Smritikosh.

Runs the memory-maintenance scheduler in its own process so the API tier can
scale horizontally without every replica running the jobs. See worker.main.
"""
