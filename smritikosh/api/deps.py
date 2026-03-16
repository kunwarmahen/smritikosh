"""
FastAPI dependency injection — shared singletons for all routes.

Pattern: @lru_cache creates each object once per process lifetime.
FastAPI's Depends() calls these functions per-request but the cache
ensures only one instance is ever created.

DB sessions (get_session, get_neo4j_session) are NOT cached — each
request gets its own session with its own transaction.
"""

from functools import lru_cache

from smritikosh.db.neo4j import get_neo4j_session  # re-exported for routes
from smritikosh.db.postgres import get_session       # re-exported for routes
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.hippocampus import Hippocampus
from smritikosh.memory.identity import IdentityBuilder
from smritikosh.memory.narrative import NarrativeMemory
from smritikosh.memory.procedural import ProceduralMemory
from smritikosh.processing.belief_miner import BeliefMiner
from smritikosh.processing.reconsolidation import ReconsolidationEngine
from smritikosh.processing.memory_clusterer import MemoryClusterer
from smritikosh.processing.reinforcement import ReinforcementLoop
from smritikosh.memory.semantic import SemanticMemory
from smritikosh.processing.amygdala import Amygdala
from smritikosh.processing.consolidator import Consolidator
from smritikosh.processing.synaptic_pruner import SynapticPruner
from smritikosh.retrieval.context_builder import ContextBuilder
from smritikosh.retrieval.intent_classifier import IntentClassifier


@lru_cache(maxsize=1)
def get_llm() -> LLMAdapter:
    """Shared LLMAdapter — one per process, reads config from settings."""
    return LLMAdapter()


@lru_cache(maxsize=1)
def get_episodic() -> EpisodicMemory:
    return EpisodicMemory()


@lru_cache(maxsize=1)
def get_semantic() -> SemanticMemory:
    return SemanticMemory()


@lru_cache(maxsize=1)
def get_amygdala() -> Amygdala:
    return Amygdala()


@lru_cache(maxsize=1)
def get_hippocampus() -> Hippocampus:
    return Hippocampus(
        llm=get_llm(),
        episodic=get_episodic(),
        semantic=get_semantic(),
        amygdala=get_amygdala(),
    )


@lru_cache(maxsize=1)
def get_narrative() -> NarrativeMemory:
    return NarrativeMemory()


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    return IntentClassifier()


@lru_cache(maxsize=1)
def get_procedural() -> ProceduralMemory:
    return ProceduralMemory()


@lru_cache(maxsize=1)
def get_context_builder() -> ContextBuilder:
    return ContextBuilder(
        llm=get_llm(),
        episodic=get_episodic(),
        semantic=get_semantic(),
        intent_classifier=get_intent_classifier(),
        narrative=get_narrative(),
        include_chains=True,
        procedural=get_procedural(),
    )


@lru_cache(maxsize=1)
def get_consolidator() -> Consolidator:
    return Consolidator(
        llm=get_llm(),
        episodic=get_episodic(),
        semantic=get_semantic(),
        narrative=get_narrative(),
    )


@lru_cache(maxsize=1)
def get_belief_miner() -> BeliefMiner:
    return BeliefMiner(llm=get_llm(), semantic=get_semantic())


@lru_cache(maxsize=1)
def get_reinforcement() -> ReinforcementLoop:
    return ReinforcementLoop()


@lru_cache(maxsize=1)
def get_clusterer() -> MemoryClusterer:
    return MemoryClusterer(llm=get_llm(), episodic=get_episodic())


@lru_cache(maxsize=1)
def get_identity_builder() -> IdentityBuilder:
    return IdentityBuilder(llm=get_llm(), semantic=get_semantic())


@lru_cache(maxsize=1)
def get_pruner() -> SynapticPruner:
    return SynapticPruner(episodic=get_episodic())


@lru_cache(maxsize=1)
def get_reconsolidation_engine() -> ReconsolidationEngine:
    return ReconsolidationEngine(llm=get_llm(), episodic=get_episodic())
