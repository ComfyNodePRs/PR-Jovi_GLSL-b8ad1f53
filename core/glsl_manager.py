"""
Jovi_GLSL - http://www.github.com/Amorano/Jovi_GLSL
GLSL Manager
"""

import gc

import glfw
from loguru import logger

from Jovi_GLSL import Singleton

# ==============================================================================
# === MANAGER ===
# ==============================================================================

def error_callback(error, description):
    logger.error(f"GLFW Error ({error}): {description}")

class GLSLManager(metaclass=Singleton):
    """GLFW initialization and global shader resources"""

    def __init__(self):
        self.__initialized = False
        glfw.set_error_callback(error_callback)

    def register_shader(self, node, shader):
        if self.__initialized == False:

            if not glfw.init():
                raise RuntimeError("GLFW failed to initialize")
            self.__active_shaders = set()
            self.__initialized = True
            logger.debug(f"GLSL Manager init")

        gc.collect()
        self.__active_shaders.add(shader)
        logger.debug(f"{node.NAME} registered")

    def unregister_shader(self, shader):
        self.__active_shaders.discard(shader)
        logger.debug(f"{shader} unregistered")
        if not self.__active_shaders and self.__initialized:
            glfw.terminate()
            self.__initialized = False
            logger.debug(f"terminated glfw")
