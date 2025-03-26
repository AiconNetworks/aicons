"""
Persistence Module for AIcon State Management

This module provides functionality for saving and loading AIcon states, including
brain state, factors, sensors, and other components to and from a PostgreSQL database.
"""

from aicons.bayesbrainGPT.persistence.persistence import AIconPersistence

__all__ = ['AIconPersistence'] 