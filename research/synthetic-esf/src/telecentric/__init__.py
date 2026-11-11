"""Telecentric silhouette simulator. See README for physical model limits."""
from .config import Camera, Illumination, Optics, Sensor, Sampling, RenderConfig
from .geometry import Primitive, Mesh, Scene
from .render import render, RenderResult
from .sensor import SensorModel

__all__ = ['Camera', 'Illumination', 'Optics', 'Sensor', 'Sampling', 'RenderConfig', 'Primitive', 'Mesh', 'Scene', 'render', 'RenderResult', 'SensorModel']
