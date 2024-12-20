"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

                            OPENGL Shaders for ComfyUI
                    http://www.github.com/Amorano/Jovi_GLSL

@title: Jovi_GLSL
@author: amorano
@category: GLSL
@reference: https://github.com/Amorano/Jovi_GLSL
@tags: GLSL, HLSL, shaders
@description: Integrates GLSL shader support.
@node list:
    GLSLNode
@version: 1.0.0
"""

import os
import sys
import json
import inspect
import importlib
from pathlib import Path

from loguru import logger

from server import PromptServer

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

ROOT = Path(__file__).resolve().parent
ROOT_COMFY = ROOT.parent.parent
ROOT_DOC = ROOT / 'res/doc'

JOV_WEB = ROOT / 'web'
JOV_INTERNAL = os.getenv("JOV_INTERNAL", 'false').strip().lower() in ('true', '1', 't')
JOV_LOG_LEVEL = os.getenv("JOV_LOG_LEVEL", "INFO")
logger.configure(handlers=[{"sink": sys.stdout, "level": JOV_LOG_LEVEL}])

# ==============================================================================
# === CORE NODES ===
# ==============================================================================

class JOVBaseNode:
    NOT_IDEMPOTENT = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ('RGBA', 'RGB', 'MASK')
    FUNCTION = "run"
    # instance map for caching
    INSTANCE = {}

    @classmethod
    def VALIDATE_INPUTS(cls, *arg, **kw) -> bool:
        # logger.debug(f'validate -- {arg} {kw}')
        return True

    @classmethod
    def INPUT_TYPES(cls, prompt:bool=False, extra_png:bool=False, dynprompt:bool=False) -> dict:
        data = {
            "required": {},
            "outputs": {
                0: ("IMAGE", {"tooltips":"Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output."}),
                1: ("IMAGE", {"tooltips":"Three channel [RGB] image. There will be no alpha."}),
                2: ("MASK", {"tooltips":"Single channel mask output."}),
            },
            "hidden": {
                "ident": "UNIQUE_ID"
            }
        }
        if prompt:
            data["hidden"]["prompt"] = "PROMPT"
        if extra_png:
            data["hidden"]["extra_pnginfo"] = "EXTRA_PNGINFO"

        if dynprompt:
            data["hidden"]["dynprompt"] = "DYNPROMPT"
        return data

class AnyType(str):
    """AnyType input wildcard trick taken from pythongossss's:

    https://github.com/pythongosssss/ComfyUI-Custom-Scripts
    """
    def __ne__(self, __value: object) -> bool:
        return False

JOV_TYPE_ANY = AnyType("*")

# want to make explicit entries; comfy only looks for single type
JOV_TYPE_COMFY = "BOOLEAN|FLOAT|INT"
JOV_TYPE_VECTOR = "VEC2|VEC3|VEC4|VEC2INT|VEC3INT|VEC4INT|COORD2D"
JOV_TYPE_NUMBER = f"{JOV_TYPE_COMFY}|{JOV_TYPE_VECTOR}"
JOV_TYPE_IMAGE = "IMAGE|MASK"
JOV_TYPE_FULL = f"{JOV_TYPE_NUMBER}|{JOV_TYPE_IMAGE}"

JOV_TYPE_COMFY = JOV_TYPE_ANY
JOV_TYPE_VECTOR = JOV_TYPE_ANY
JOV_TYPE_NUMBER = JOV_TYPE_ANY
JOV_TYPE_IMAGE = JOV_TYPE_ANY
JOV_TYPE_FULL = JOV_TYPE_ANY

GLSL_INTERNAL = '🧙🏽'
GLSL_CUSTOM = '🧙🏽‍♀️'

# ==============================================================================
# === NODE LOADER ===
# ==============================================================================

def comfy_message(ident:str, route:str, data:dict) -> None:
    data['id'] = ident
    PromptServer.instance.send_sync(route, data)

# ==============================================================================
# === NODE LOADER ===
# ==============================================================================

def loader():
    node_count = 0
    CLASS_MAPPINGS = {}
    CLASS_MAPPINGS_WIP = {}
    NODE_LIST_MAP = {}
    for fname in ROOT.glob('core/**/*.py'):
        if fname.stem.startswith('_'):
            continue

        try:
            route = str(fname).replace("\\", "/").split("Jovi_GLSL/core/")[1]
            route = route.split('.')[0].replace('/', '.')
            module = f"Jovi_GLSL.core.{route}"
            module = importlib.import_module(module)
        except Exception as e:
            logger.warning(f"module failed {fname}")
            logger.warning(str(e))
            continue

        # check if there is a dynamic register function....
        try:
            for class_name, class_def in module.import_dynamic():
                setattr(module, class_name, class_def)
                logger.debug(f"shader: {class_name}")
        except Exception as e:
            pass

        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, class_object in classes:
            # assume both attrs are good enough....
            if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                name = class_object.NAME
                CLASS_MAPPINGS[name] = class_object
                if not name.endswith(GLSL_CUSTOM):
                    desc = class_object.DESCRIPTION if hasattr(class_object, 'DESCRIPTION') else name
                    NODE_LIST_MAP[name] = desc.split('.')[0].strip('\n')
                else:
                    logger.debug(f"customs {name}")
                node_count += 1

        logger.info(f"✅ {module.__name__}")
    logger.info(f"{node_count} nodes loaded")

    global NODE_DISPLAY_NAME_MAPPINGS, NODE_CLASS_MAPPINGS

    NODE_DISPLAY_NAME_MAPPINGS = {k: v.NAME_PRETTY if hasattr(v, 'NAME_PRETTY') else k for k, v in CLASS_MAPPINGS.items()}
    CLASS_MAPPINGS.update({k: v for k, v in CLASS_MAPPINGS_WIP.items()})
    NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k in CLASS_MAPPINGS_WIP.keys()})
    CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(CLASS_MAPPINGS.items(),
                                                            key=lambda item: getattr(item[1], 'SORT', 0))}

    for k, v in CLASS_MAPPINGS.items():
        NODE_CLASS_MAPPINGS[k] = v

    # only do the list on local runs...
    if JOV_INTERNAL:
        with open(str(ROOT) + "/node_list.json", "w", encoding="utf-8") as f:
            json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

loader()
