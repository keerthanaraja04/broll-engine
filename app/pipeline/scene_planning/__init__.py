"""
Scene planning package.

Converts a Transcript into a list of PlannedScene objects,
each with b-roll search queries and semantic metadata.
"""

from app.pipeline.scene_planning.models import PlannedScene, ScenePlan
from app.pipeline.scene_planning.planner import plan_scenes
from app.pipeline.scene_planning.scene_step import ScenePlanningStep

__all__ = ["PlannedScene", "ScenePlan", "plan_scenes", "ScenePlanningStep"]
