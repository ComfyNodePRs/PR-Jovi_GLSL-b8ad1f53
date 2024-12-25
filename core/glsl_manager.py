"""
Jovi_GLSL - http://www.github.com/Amorano/Jovi_GLSL
GLSL Manager
"""

import glfw
from loguru import logger

from Jovi_GLSL import Singleton

# ==============================================================================
# === MANAGER ===
# ==============================================================================

class GLSLManager(metaclass=Singleton):
    """GLFW initialization and global shader resources"""

    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW failed to initialize")
        self.__active_shaders = set()
        logger.debug(f"GLSL Manager init")

    def register_shader(self, node, shader):
        self.__active_shaders.add(shader)
        logger.debug(f"{node.NAME} registered")

    def unregister_shader(self, shader):
        self.__active_shaders.discard(shader)
        # logger.debug(f"{shader} unregistered")
        if not self.__active_shaders:
            glfw.terminate()
