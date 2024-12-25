"""
Jovi_GLSL - http://www.github.com/Amorano/Jovi_GLSL
Vector supports
"""

import sys

from Jovi_GLSL import JOVBaseNode

class VECTOR2Node(JOVBaseNode):
    NAME = "VECTOR2 FLOAT (JOV_GL)"
    RETURN_TYPES = ("VEC2", )
    RETURN_NAMES = ("VEC2", )
    DESCRIPTION = """

"""
    SORT = 10030

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
                "y": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
            }
        }

    def run(self, x, y):
        return ((x, y), )

class VECTOR2INode(JOVBaseNode):
    NAME = "VECTOR2 INTEGER (JOV_GL)"
    RETURN_TYPES = ("VEC2I", )
    RETURN_NAMES = ("VEC2I", )
    DESCRIPTION = """

"""
    SORT = 10035

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("INT", { "default": 0, "step": 1, "min": -sys.maxsize, "max": sys.maxsize }),
                "y": ("INT", { "default": 0, "step": 1, "min": -sys.maxsize, "max": sys.maxsize }),
            }
        }

    def run(self, x, y):
        return ((x, y), )

class VECTOR3Node(JOVBaseNode):
    NAME = "VECTOR3 FLOAT (JOV_GL)"
    RETURN_TYPES = ("VEC3", )
    RETURN_NAMES = ("VEC3", )
    DESCRIPTION = """

"""
    SORT = 10040

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
                "y": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
                "z": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
            }
        }

    def run(self, x, y, z):
        return ((x, y, z), )

class VECTOR3INode(JOVBaseNode):
    NAME = "VECTOR3 INTEGER (JOV_GL)"
    RETURN_TYPES = ("VEC3", )
    RETURN_NAMES = ("VEC3", )
    DESCRIPTION = """

"""
    SORT = 10045

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("INT", { "default": 0, "step": 1, "min": -sys.maxsize, "max": sys.maxsize }),
                "y": ("INT", { "default": 0, "step": 1, "min": -sys.maxsize, "max": sys.maxsize }),
                "z": ("INT", { "default": 0, "step": 1, "min": -sys.maxsize, "max": sys.maxsize }),
            }
        }

    def run(self, x, y, z):
        return ((x, y, z), )

class VECTOR4Node(JOVBaseNode):
    NAME = "VECTOR4 FLOAT (JOV_GL)"
    RETURN_TYPES = ("VEC4", )
    RETURN_NAMES = ("VEC4", )
    DESCRIPTION = """

"""
    SORT = 10050

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
                "y": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
                "z": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
                "w": ("FLOAT", { "default": 0.0, "step": 0.01, "min": -sys.maxsize, "max": sys.maxsize }),
            }
        }

    def run(self, x, y, z, w):
        return ((x, y, z, w), )

class RGBAFloatNode(JOVBaseNode):
    NAME = "RGBA FLOAT (JOV_GL)"
    RETURN_TYPES = ("VEC4", )
    RETURN_NAMES = ("VEC4", )
    DESCRIPTION = """

"""
    SORT = 10060

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r": ("FLOAT", { "default": 0.0, "step": 0.01 }),
                "g": ("FLOAT", { "default": 0.0, "step": 0.01 }),
                "b": ("FLOAT", { "default": 0.0, "step": 0.01 }),
                "a": ("FLOAT", { "default": 0.0, "step": 0.01 }),
            }
        }

    def run(self, r, g, b, a):
        return ((r, g, b, a), )

class RGBAIntegerNode(JOVBaseNode):
    NAME = "RGBA INTEGER (JOV_GL)"
    RETURN_TYPES = ("VEC4", )
    RETURN_NAMES = ("VEC4", )
    DESCRIPTION = """

"""
    SORT = 10065

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r": ("INT", { "default": 0, "step": 1, "min": 0, "max": 255 }),
                "g": ("INT", { "default": 0, "step": 1, "min": 0, "max": 255 }),
                "b": ("INT", { "default": 0, "step": 1, "min": 0, "max": 255 }),
                "a": ("INT", { "default": 0, "step": 1, "min": 0, "max": 255 }),
            }
        }

    def run(self, r, g, b, a):
        return ((r, g, b, a), )
