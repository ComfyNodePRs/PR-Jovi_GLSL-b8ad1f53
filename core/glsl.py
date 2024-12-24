"""
Jovi_GLSL - http://www.github.com/Amorano/Jovi_GLSL
Creation
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
from enum import Enum, EnumMeta as EnumType

import cv2
import glfw
import torch
import numpy as np
import OpenGL.GL as gl
from loguru import logger

from comfy.utils import ProgressBar

from Jovi_GLSL import GLSL_INTERNAL, GLSL_CUSTOM, JOV_TYPE_IMAGE, ROOT, \
    JOVBaseNode, \
    comfy_message

from Jovi_GLSL.core import IMAGE_SIZE_DEFAULT, IMAGE_SIZE_MAX, IMAGE_SIZE_MIN, \
    EnumConvertType, \
    cv2tensor_full, image_convert, load_file, \
    parse_param, parse_value, tensor2cv

# ==============================================================================
# === SHADER LOADER ===
# ==============================================================================

ROOT_GLSL = ROOT / 'glsl'
GLSL_PROGRAMS = {
    "vertex": {  },
    "fragment": { }
}

GLSL_PROGRAMS['vertex'].update({str(f.relative_to(ROOT_GLSL).as_posix()):
                                str(f) for f in Path(ROOT_GLSL).rglob('*.vert')})

USER_GLSL = ROOT / '_user'
USER_GLSL.mkdir(parents=True, exist_ok=True)
if (USER_GLSL := os.getenv("JOV_GLSL", str(USER_GLSL))) is not None:
    GLSL_PROGRAMS['vertex'].update({str(f.relative_to(USER_GLSL).as_posix()):
                                    str(f) for f in Path(USER_GLSL).rglob('*.vert')})

GLSL_PROGRAMS['fragment'].update({str(f.relative_to(ROOT_GLSL).as_posix()):
                                  str(f) for f in Path(ROOT_GLSL).rglob('*.frag')})
if USER_GLSL is not None:
    GLSL_PROGRAMS['fragment'].update({str(f.relative_to(USER_GLSL).as_posix()):
                                      str(f) for f in Path(USER_GLSL).rglob('*.frag')})

try:
    prog = GLSL_PROGRAMS['vertex'].pop('.lib/_.vert')
    PROG_VERTEX = load_file(prog)
except Exception as e:
    logger.error(e)
    raise Exception("failed load default vertex program .lib/_.vert")

try:
    prog = GLSL_PROGRAMS['fragment'].pop('.lib/_.frag')
    PROG_FRAGMENT = load_file(prog)
except Exception as e:
    logger.error(e)
    raise Exception("failed load default fragment program .lib/_.frag")

PROG_HEADER = load_file(ROOT_GLSL / '.lib/_.head')
PROG_FOOTER = load_file(ROOT_GLSL / '.lib/_.foot')

logger.info(f"  vertex programs: {len(GLSL_PROGRAMS['vertex'])}")
logger.info(f"fragment programs: {len(GLSL_PROGRAMS['fragment'])}")

# ==============================================================================
# === CONSTANT ===
# ==============================================================================

RE_INCLUDE = re.compile(r"^\s+?#include\s+?([A-Za-z\_\-\.\\\/]{3,})$", re.MULTILINE)
RE_VARIABLE = re.compile(r"uniform\s+(\w+)\s+(\w+);\s*(?:\/\/\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:\|\s*(.*))?$", re.MULTILINE)
RE_SHADER_META = re.compile(r"^\/\/\s?([A-Za-z_]{3,}):\s?(.+)$", re.MULTILINE)

# HALFPI: float = math.pi / 2
# TAU: float = math.pi * 2

LAMBDA_UNIFORM = {
    'bool': gl.glUniform1i,
    'int': gl.glUniform1i,
    'ivec2': gl.glUniform2i,
    'ivec3': gl.glUniform3i,
    'ivec4': gl.glUniform4i,
    'float': gl.glUniform1f,
    'vec2': gl.glUniform2f,
    'vec3': gl.glUniform3f,
    'vec4': gl.glUniform4f,
}

PTYPE = {
    'bool': EnumConvertType.BOOLEAN,
    'int': EnumConvertType.INT,
    'ivec2': EnumConvertType.VEC2INT,
    'ivec3': EnumConvertType.VEC3INT,
    'ivec4': EnumConvertType.VEC4INT,
    'float': EnumConvertType.FLOAT,
    'vec2': EnumConvertType.VEC2,
    'vec3': EnumConvertType.VEC3,
    'vec4': EnumConvertType.VEC4,
    'sampler2D': EnumConvertType.IMAGE
}

# ==============================================================================
# === ENUMERTATION ===
# ==============================================================================

class EnumEdgeWrap(Enum):
    CLAMP  = 10
    WRAP   = 20
    MIRROR = 30

# ==============================================================================
# === SHADER ENUMERTATION ===
# ==============================================================================

"""
Enumerations exposed to the shader scripts
"""
class EnumGLSLColorConvert(Enum):
    RGB2HSV = 0
    RGB2LAB = 1
    RGB2XYZ = 2
    HSV2RGB = 10
    HSV2LAB = 11
    HSV2XYZ = 12
    LAB2RGB = 20
    LAB2HSV = 21
    LAB2XYZ = 22
    XYZ2RGB = 30
    XYZ2HSV = 31
    XYZ2LAB = 32

# ==============================================================================
# === SHADER SUPPORT ===
# ==============================================================================

class CompileException(Exception): pass

class GLSLManager:
    """Singleton manager for GLFW initialization and global shader resources"""
    _instance = None
    _init = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._init:
            return
        if not glfw.init():
            raise RuntimeError("GLFW failed to initialize")
        self._init = True
        self.__active_shaders = set()

    def register_shader(self, shader):
        self.__active_shaders.add(shader)

    def unregister_shader(self, shader):
        self.__active_shaders.discard(shader)
        if not self.__active_shaders and self._init:
            glfw.terminate()
            self._init = False

class GLSLShader:
    """
    """

    def __init__(self, vertex:str=None, fragment:str=None, width:int=IMAGE_SIZE_DEFAULT, height:int=IMAGE_SIZE_DEFAULT, fps:int=30) -> None:
        self.__glsl_manager = GLSLManager()
        self.__glsl_manager.register_shader(self)

        self.__size: Tuple[int, int] = (max(width, IMAGE_SIZE_MIN), max(height, IMAGE_SIZE_MIN))
        self.__runtime: float = 0
        self.__fps: int = min(120, max(1, fps))
        self.__mouse: Tuple[int, int] = (0, 0)
        self.__bgcolor = (0, 0, 0, 1.)
        self.__textures = {}
        self.__uniform_state = {}
        self.__texture_hashes = {}

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.__window = glfw.create_window(self.__size[0], self.__size[1], "hidden", None, None)
        if not self.__window:
            # glfw.terminate()
            raise RuntimeError("GLFW did not init window")
        logger.debug("window created")

        glfw.make_context_current(self.__window)

        # framebuffer
        self.__fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        logger.debug("framebuffer created")

        self.__fbo_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.__fbo_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F,
                        self.__size[0], self.__size[1], 0,
                        gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.__fbo_texture, 0)
        logger.debug("framebuffer texture created")

        gl.glViewport(0, 0, self.__size[0], self.__size[1])
        logger.debug("viewport created")

        self.__empty_image: np.ndarray = np.zeros((self.__size[0], self.__size[1]), np.uint8)
        self.__last_frame: np.ndarray = np.zeros((self.__size[0], self.__size[1]), np.uint8)

        try:
            if vertex is None:
                logger.debug("vertex program is empty. using default.")
                vertex = PROG_VERTEX
            vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
            gl.glShaderSource(vertex_shader, vertex)
            gl.glCompileShader(vertex_shader)
            if gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
                raise CompileException(gl.glGetShaderInfoLog(vertex_shader).decode())
            logger.debug("vertex program ready")

            if fragment is None:
                logger.debug("fragment program is empty. using default.")
                fragment = PROG_FRAGMENT
            fragment_raw = PROG_HEADER + fragment + PROG_FOOTER
            fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
            gl.glShaderSource(fragment_shader, fragment_raw)
            gl.glCompileShader(fragment_shader)
            if gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
                gl.glDeleteShader(vertex_shader)
                raise CompileException(gl.glGetShaderInfoLog(fragment_shader).decode())
            logger.debug("fragment program ready")

            self.__program = gl.glCreateProgram()
            gl.glAttachShader(self.__program, vertex_shader)
            gl.glAttachShader(self.__program, fragment_shader)
            gl.glLinkProgram(self.__program)
            logger.debug("program linked")

            # Clean up shaders after linking
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)

            if gl.glGetProgramiv(self.__program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
                raise CompileException(gl.glGetProgramInfoLog(self.__program).decode())

            gl.glUseProgram(self.__program)

            # Setup uniforms
            self.__shaderVar = {}
            statics = ['iResolution', 'iTime', 'iFrameRate', 'iFrame']
            for s in statics:
                if (val := gl.glGetUniformLocation(self.__program, s)) > -1:
                    self.__shaderVar[s] = val

            if (resolution := self.__shaderVar.get('iResolution', -1)) > -1:
                gl.glUniform3f(resolution, self.__size[0], self.__size[1], 0)
            logger.debug("uniforms initialized")

            # Setup user variables
            self.__userVar = {}
            for match in RE_VARIABLE.finditer(fragment):
                typ, name, default, *_ = match.groups()

                texture = None
                if typ in ['sampler2D']:
                    texture = self.__textures[name] = gl.glGenTextures(1)
                else:
                    default = default.strip()
                    if default.startswith('EnumGLSL'):
                        typ = 'int'
                        default = getattr(sys.modules[__name__], default, 0)

                self.__userVar[name] = [typ, gl.glGetUniformLocation(self.__program, name), default, texture]

            logger.debug("user uniforms initialized")
        except Exception as e:
            self.__cleanup()
            raise CompileException(f"shader compilation failed: {str(e)}")

    def __update_framebuffer_size(self) -> None:
        """Update framebuffer and texture sizes without recreating them"""
        glfw.make_context_current(self.__window)
        glfw.set_window_size(self.__window, self.__size[0], self.__size[1])

        # Bind existing FBO and texture
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.__fbo_texture)

        # Update sizes
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F,
                       self.__size[0], self.__size[1], 0,
                       gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glViewport(0, 0, self.__size[0], self.__size[1])

        # Update internal buffers
        if self.__empty_image.shape != (self.__size[0], self.__size[1]):
            self.__empty_image = np.zeros((self.__size[0], self.__size[1]), np.uint8)
            self.__last_frame = np.zeros((self.__size[0], self.__size[1]), np.uint8)

        # Clear texture hashes to force texture updates at new size
        self.__texture_hashes.clear()

    def __cleanup(self):
        """Explicit cleanup of OpenGL resources"""
        if hasattr(self, '_cleanup_called'):
            return
        self._cleanup_called = True

        if self.__window:
            glfw.make_context_current(self.__window)

            texture_ids = [v[3] for v in self.__userVar.values() if v[0] == 'sampler2D']
            if texture_ids:
                gl.glDeleteTextures(texture_ids)

            if self.__fbo_texture:
                gl.glDeleteTextures([self.__fbo_texture])
                self.__fbo_texture = None

            if self.__fbo:
                gl.glDeleteFramebuffers([self.__fbo])
                self.__fbo = None

            if self.__program:
                gl.glDeleteProgram(self.__program)
                self.__program = None

            glfw.destroy_window(self.__window)
            self.__glsl_manager.unregister_shader(self)
            self.__window = None

    def __del__(self):
        """Cleanup during garbage collection"""
        try:
            self.__cleanup()
        except:
            pass

    @property
    def size(self) -> Tuple[int, int]:
        return self.__size

    @size.setter
    def size(self, size: Tuple[int, int]) -> None:
        sz = (min(IMAGE_SIZE_MAX, max(IMAGE_SIZE_MIN, size[0])),
              min(IMAGE_SIZE_MAX, max(IMAGE_SIZE_MIN, size[1])))

        if sz[0] != self.__size[0] or sz[1] != self.__size[1]:
            self.__size = sz
            self.__update_framebuffer_size()
            logger.debug(f"size set {sz} ==> {self.__size}")

            # Update resolution uniform
            if (rez := self.__shaderVar.get('iResolution', -1)) > -1:
                glfw.make_context_current(self.__window)
                gl.glUseProgram(self.__program)
                gl.glUniform3f(rez, self.__size[0], self.__size[1], 0)

    @property
    def runtime(self) -> float:
        return self.__runtime

    @runtime.setter
    def runtime(self, runtime:float) -> None:
        runtime = max(0, runtime)
        self.__runtime = runtime

    @property
    def fps(self) -> int:
        return self.__fps

    @fps.setter
    def fps(self, fps:int) -> None:
        fps = max(1, min(120, int(fps)))
        self.__fps = fps
        if (iFrameRate := self.__shaderVar.get('iFrameRate', -1)) > -1:
            glfw.make_context_current(self.__window)
            gl.glUseProgram(self.__program)
            gl.glUniform1f(self.__shaderVar['iFrameRate'], iFrameRate)

    @property
    def mouse(self) -> Tuple[int, int]:
        return self.__mouse

    @mouse.setter
    def mouse(self, pos:Tuple[int, int]) -> None:
        self.__mouse = pos

    @property
    def frame(self) -> float:
        return int(self.__runtime * self.__fps)

    @property
    def last_frame(self) -> float:
        return self.__last_frame

    @property
    def bgcolor(self) -> Tuple[int, ...]:
        return self.__bgcolor

    @bgcolor.setter
    def bgcolor(self, color:Tuple[int, ...]) -> None:
        self.__bgcolor = tuple(float(x) / 255. for x in color)

    def render(self, time_delta:float=0.,
               tile_edge:Tuple[EnumEdgeWrap,...]=(EnumEdgeWrap.CLAMP, EnumEdgeWrap.CLAMP),
               **kw) -> np.ndarray:

        glfw.make_context_current(self.__window)
        gl.glUseProgram(self.__program)

        self.runtime = time_delta

        # current time in shader lifetime
        if (val := self.__shaderVar.get('iTime', -1)) > -1:
            gl.glUniform1f(val, self.__runtime)

        # the desired FPS
        if (val := self.__shaderVar.get('iFrameRate', -1)) > -1:
            gl.glUniform1i(val, self.__fps)

        # the current frame based on the life time and "fps"
        if (val := self.__shaderVar.get('iFrame', -1)) > -1:
            gl.glUniform1i(val, self.frame)

        texture_index = -1

        for uk, uv in self.__userVar.items():
            p_type, p_loc, p_value, texture_id = uv
            val = kw.get(uk, p_value)

            if p_type == 'sampler2D':
                texture_index += 1
                if texture_id is None:
                    continue
                    if (texture := self.__textures.get(uk, None)) is None:
                        logger.error(f"texture [{texture_index}] {uk} is None")
                        texture_index += 1
                        continue

                gl.glActiveTexture(gl.GL_TEXTURE0 + texture_index)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

                if isinstance(val, np.ndarray):
                    current_hash = hash((val.ctypes.data, val.shape, val.dtype))
                else:
                    current_hash = hash(0)

                if uk not in self.__texture_hashes or self.__texture_hashes[uk] != current_hash:
                    val = image_convert(val, 4)
                    val = val[::-1,:]
                    val = val.astype(np.float32) / 255.0
                    val = cv2.resize(val, self.__size, interpolation=cv2.INTER_LINEAR)

                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.__size[0], self.__size[1], 0, gl.GL_RGBA, gl.GL_FLOAT, val)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

                    self.__texture_hashes[uk] = current_hash

                    # Set edge wrapping modes
                    for idx, text_wrap in enumerate([gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T]):
                        if tile_edge[idx] == EnumEdgeWrap.WRAP:
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, text_wrap, gl.GL_REPEAT)
                        elif tile_edge[idx] == EnumEdgeWrap.MIRROR:
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, text_wrap, gl.GL_MIRRORED_REPEAT)
                        else:
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, text_wrap, gl.GL_CLAMP_TO_EDGE)

                gl.glUniform1i(p_loc, texture_index)
            elif val:
                if isinstance(p_value, EnumType):
                    val = p_value[val].value
                elif isinstance(val, str):
                    val = val.split(',')
                val = parse_value(val, PTYPE[p_type], 0)
                if not isinstance(val, (list, tuple)):
                    val = [val]

                uk = (uk, p_loc)
                if uk not in self.__uniform_state or self.__uniform_state[uk] != val:
                    LAMBDA_UNIFORM[p_type](p_loc, *val)
                    self.__uniform_state[uk] = val

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        gl.glClearColor(*self.__bgcolor)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

        data = gl.glReadPixels(0, 0, self.__size[0], self.__size[1], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.__size[1], self.__size[0], 4)
        self.__last_frame = image[::-1, :, :]

        glfw.poll_events()
        return self.__last_frame

def shader_meta(shader: str) -> Dict[str, Any]:
    ret = {}
    for match in RE_SHADER_META.finditer(shader):
        key, value = match.groups()
        ret[key] = value
    ret['_'] = [match.groups() for match in RE_VARIABLE.finditer(shader)]
    return ret

def load_file_glsl(fname: str) -> str:

    # first file we load, starts the list of included
    include = set()

    def scan_include(file:str, idx:int=0) -> str:
        if idx > 4:
            return "too many recursive includes"

        file_path = ROOT_GLSL / file
        if file_path in include:
            return ""

        include.add(file_path)
        try:
            result = load_file(file_path)
        except FileNotFoundError:
            return f"File not found: {file_path}"

        # replace #include directives with their content
        def replace_include(match):
            lib_path = ROOT_GLSL / match.group(1)
            if lib_path not in include:
                return scan_include(lib_path, idx+1)
            return ""

        return RE_INCLUDE.sub(replace_include, result)

    return scan_include(fname)

# ==============================================================================
# === COMFYUI NODES ===
# ==============================================================================

class GLSLNodeDynamic(JOVBaseNode):
    CATEGORY = f"JOVI_GLSL 🔺🟩🔵"
    CONTROL = []
    PARAM = []

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        original_params = super().INPUT_TYPES()
        for ctrl in cls.CONTROL:
            match ctrl.upper():
                case "WH":
                    original_params["optional"]["WH"] = ("VEC2INT", {"default": (512, 512), "mij":IMAGE_SIZE_MIN, "label": ['W', 'H']})
                case "MATTE":
                    original_params["optional"]["MATTE"] = ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
        """
        'MODE': (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name})
        'SAMPLE': (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name})
        'EDGE_X': (EnumEdgeWrap._member_names_, {"default": EnumEdgeWrap.CLAMP.name})
        'EDGE_Y': (EnumEdgeWrap._member_names_, {"default": EnumEdgeWrap.CLAMP.name})
        """

        opts = original_params.get('optional', {})
        opts.update({
            'FRAGMENT': ("STRING", {"default": cls.FRAGMENT}),
        })

        # parameter list first...
        data = {}
        # if cls.PARAM is not None:
        # 1., 1., 1.; 0; 1; 0.01; rgb | End of the Range
        # default, min, max, step, metadata, tooltip
        for glsl_type, name, default, val_min, val_max, val_step, meta, tooltip in cls.PARAM:
            typ = PTYPE[glsl_type]
            params = {"default": None}

            d = None
            type_name = JOV_TYPE_IMAGE
            if glsl_type != 'sampler2D':
                type_name = typ.name
                if default is not None:
                    if default.startswith('EnumGLSL'):
                        if (target_enum := globals().get(default.strip(), None)) is not None:
                            # this be an ENUM....
                            type_name = target_enum._member_names_
                            params['default'] = type_name[0]
                        else:
                            params['default'] = 0
                    else:
                        d = default.split(',')
                        params['default'] = parse_value(d, typ, 0)

                def minmax(mm: str, what: str) -> str:
                    match glsl_type:
                        case 'int'|'float':
                            mm = what
                    return mm

                if val_min is not None:
                    if val_min == "":
                        val_min = -sys.maxsize
                    params[minmax('mij', 'min')] = parse_value(val_min, EnumConvertType.FLOAT, -sys.maxsize)

                if val_max is not None:
                    if val_max == "":
                        val_max = sys.maxsize
                    params[minmax('maj', 'max')] = parse_value(val_max, EnumConvertType.FLOAT, sys.maxsize)

                if val_step is not None:
                    d = 1 if typ.name.endswith('INT') else 0.01
                    params['step'] = parse_value(val_step, EnumConvertType.FLOAT, d)

                if meta is not None:
                    if "rgb" in meta:
                        if glsl_type.startswith('vec'):
                            params['linear'] = True
                        else:
                            params['rgb'] = True

            if tooltip is not None:
                params["tooltip"] = tooltip
            data[name] = (type_name, params,)

        data.update(opts)
        original_params['optional'] = data
        return original_params

    #@classmethod
    #def IS_CHANGED(cls, **kw) -> float:
    #    return float('nan')

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = None
        self.__delta = 0

    def run(self, ident, **kw) -> Tuple[torch.Tensor]:
        batch = parse_param(kw, 'BATCH', EnumConvertType.INT, 0, 0, 1048576)[0]
        delta = parse_param(kw, 'TIME', EnumConvertType.FLOAT, 0)[0]

        # everybody wang comp tonight
        #mode = parse_param(kw, 'MODE', EnumScaleMode, EnumScaleMode.MATTE.name)[0]
        wihi = parse_param(kw, 'WH', EnumConvertType.VEC2INT, [(512, 512)], IMAGE_SIZE_MIN)[0]
        #sample = parse_param(kw, 'SAMPLE', EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
        matte = parse_param(kw, 'MATTE', EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)[0]
        #edge_x = parse_param(kw, 'EDGE_X', EnumEdgeWrap, EnumEdgeWrap.CLAMP.name)[0]
        #edge_y = parse_param(kw, 'EDGE_Y', EnumEdgeWrap, EnumEdgeWrap.CLAMP.name)[0]
        #edge = (edge_x, edge_y)

        variables = kw.copy()
        for p in ['WH', 'MATTE', 'BATCH', 'TIME', 'FPS']:
            variables.pop(p, None)

        if self.__glsl is None:
            try:
                vertex = getattr(self, 'VERTEX', kw.pop('VERTEX', None))
                fragment = getattr(self, 'FRAGMENT', kw.pop('FRAGMENT', None))
            except CompileException as e:
                comfy_message(ident, "jovi-glsl-error", {"id": ident, "e": str(e)})
                logger.error(self.NAME)
                logger.error(e)
                return

            fps = parse_param(kw, 'FPS', EnumConvertType.INT, 24, 1, 120)[0]
            self.__glsl = GLSLShader(vertex, fragment, fps=fps)

        if batch > 0 or self.__delta != delta:
            self.__delta = delta
        step = 1. / self.__glsl.fps

        images = []
        vars = {}
        batch = max(1, batch)

        for k, var in variables.items():
            if isinstance(var, (torch.Tensor)):
                batch = max(batch, var.shape[0])
                var = [image_convert(tensor2cv(v), 4) for v in var]
            elif isinstance(var, (list, tuple,)):
                batch = max(batch, len(var))
            variables[k] = var if isinstance(var, (list, tuple,)) else [var]

        # if there are input images, use the first one we come across as the w,h requirement
        # unless there is an explcit override?
        firstImage = None
        pbar = ProgressBar(batch)
        for idx in range(batch):
            for k, val in variables.items():
                vars[k] = val[idx % len(val)]
                if firstImage is None and 'WH' not in self.CONTROL and isinstance(vars[k], (np.ndarray,)):
                    firstImage = vars[k].shape[:2][::-1]

            self.__glsl.size = wihi if firstImage is None else firstImage

            img = self.__glsl.render(self.__delta, **vars)
            images.append(cv2tensor_full(img, matte))
            self.__delta += step
            comfy_message(ident, "jovi-glsl-time", {"id": ident, "t": self.__delta})
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

def import_dynamic() -> Tuple[str,...]:
    ret = []
    sort = 10000
    root = str(ROOT_GLSL)
    for name, fname in GLSL_PROGRAMS['fragment'].items():
        if (shader := load_file_glsl(fname)) is None:
            logger.error(f"missing shader file {fname}")
            continue

        meta = shader_meta(shader)
        if meta.get('hide', False):
            continue

        name = meta.get('name', name.split('.')[0]).upper()
        class_name = name.title().replace(' ', '_')
        class_name = f'GLSLNode_{class_name}'

        emoji = GLSL_CUSTOM
        sort_order = sort
        if fname.startswith(root):
            emoji = GLSL_INTERNAL
            sort_order -= 10000

        category = GLSLNodeDynamic.CATEGORY
        if (sub := meta.get('category', None)) is not None:
            category += f'/{sub}'

        class_def = type(class_name, (GLSLNodeDynamic,), {
            "NAME": f'GLSL {name} (JOV_GL) {emoji}'.upper(),
            "DESCRIPTION": meta.get('desc', name),
            "CATEGORY": category.upper(),
            "FRAGMENT": shader,
            "PARAM": meta.get('_', []),
            "CONTROL": [x.upper().strip() for x in meta.get('control', "").split(",") if len(x) > 0],
            "SORT": sort_order,
        })

        sort += 10
        ret.append((class_name, class_def,))
    return ret
