"""
Jovi_GLSL - http://www.github.com/Amorano/Jovi_GLSL
GLSL
"""

import re
import sys
from typing import Any, Dict, Tuple
from enum import Enum

import torch
from loguru import logger

from comfy.utils import ProgressBar

from Jovi_GLSL import GLSL_INTERNAL, GLSL_CUSTOM, JOV_TYPE_IMAGE, \
    comfy_message, load_file, zip_longest_fill

from Jovi_GLSL.core import GLSL_PROGRAMS, PTYPE, RE_VARIABLE, ROOT_GLSL, \
    IMAGE_SIZE_MIN, IMAGE_SIZE_DEFAULT, IMAGE_SIZE_MAX, \
    CompileException, JOVBaseGLSLNode, EnumConvertType, \
    cv2tensor_full, image_convert, parse_param, parse_value, tensor2cv

from Jovi_GLSL.core.glsl_shader import GLSLShader

# ==============================================================================
# === CONSTANT ===
# ==============================================================================

RE_INCLUDE = re.compile(r"^\s*#include\s+([A-Za-z_\-\.\\\/]{3,})$", re.MULTILINE)
RE_SHADER_META = re.compile(r"^\/\/\s?([A-Za-z_]{3,}):\s?(.+)$", re.MULTILINE)

# ==============================================================================
# === ENUMERTATION ===
# ==============================================================================

class EnumEdgeWrap(Enum):
    CLAMP  = 10
    WRAP   = 20
    MIRROR = 30

# ==============================================================================
# === COMFYUI NODE ===
# ==============================================================================

class GLSLNodeDynamic(JOVBaseGLSLNode):
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
        # current frame, if we are in batch mode, this will advance
        self.__frame = 0

    def run(self, ident, **kw) -> Tuple[torch.Tensor]:
        # batch is a single value entry -- drives everyone else.
        batch = parse_param(kw, 'BATCH', EnumConvertType.INT, 0, 0, 1048576)[0]
        batch = [batch] * max(1, batch)

        iFrame = parse_param(kw, 'FRAME', EnumConvertType.FLOAT, 0)
        iFrameRate = parse_param(kw, 'FPS', EnumConvertType.INT, 24)
        iResolution = parse_param(kw, 'WH', EnumConvertType.VEC2INT,
                                  [(IMAGE_SIZE_DEFAULT, IMAGE_SIZE_DEFAULT)],
                                  IMAGE_SIZE_MIN, IMAGE_SIZE_MAX)
        bgcolor = parse_param(kw, 'MATTE', EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)

        #edge_x = parse_param(kw, 'EDGE_X', EnumEdgeWrap, EnumEdgeWrap.CLAMP.name)[0]
        #edge_y = parse_param(kw, 'EDGE_Y', EnumEdgeWrap, EnumEdgeWrap.CLAMP.name)[0]
        #edge = (edge_x, edge_y)

        variables = kw.copy()
        for k in ['BATCH', 'FRAME', 'FPS', 'WH', 'MATTE']:
            variables.pop(k, None)

        if self.__glsl is None:
            try:
                vertex = getattr(self, 'VERTEX', kw.pop('VERTEX', None))
                fragment = getattr(self, 'FRAGMENT', kw.pop('FRAGMENT', None))
            except Exception as e:
                comfy_message(ident, "jovi-glsl-error", {"id": ident, "e": str(e)})
                logger.error(self.NAME)
                logger.error(e)
                return

            self.__glsl = GLSLShader(self, vertex, fragment)

        for k, var in variables.items():
            variables[k] = var if isinstance(var, (list, )) else [var]

        # if the batch == 0 then we want an automagic frame step
        if batch == 0:
            start_frame = iFrame[0]
            for x in range(len(batch)):
                iFrame.append(start_frame + x)
            iTime = [frame/rate for (frame, rate) in list(zip_longest_fill(iFrame, iFrameRate))]
        else:
            iTime = [iFrame[0] / iFrameRate[0]]

        images = []
        params = list(zip_longest_fill(batch, iTime, iFrame, iResolution, bgcolor))
        pbar = ProgressBar(len(params))
        for idx, (batch, iTime, iFrame, iResolution, bgcolor) in enumerate(params):
            vars = {}
            firstImage = None
            for k, val in variables.items():
                vars[k] = val[idx % len(val)]
                # convert images, grab first one if no sizes provided
                if isinstance(vars[k], (torch.Tensor,)):
                    vars[k] = vars[k][idx % len(val)]
                    vars[k] = image_convert(tensor2cv(vars[k]), 4)
                    if firstImage is None and 'WH' not in self.CONTROL:
                        firstImage = True #vars[k].shape[:2][::-1]
                        iResolution = vars[k].shape[:2][::-1]

            vars['bgcolor'] = bgcolor

            img = self.__glsl.render(iTime, iFrame, iResolution, **vars)
            images.append(cv2tensor_full(img, bgcolor))

            # self.__delta += self.__glsl.step
            comfy_message(ident, "jovi-glsl-time", {"id": ident, "t": self.__delta})
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        if idx > 8:
            raise CompileException(f"too many file include recursions ({idx})")

        file_path = ROOT_GLSL / file
        if file_path in include:
            return ""

        include.add(file_path)
        try:
            result = load_file(file_path)
        except FileNotFoundError:
            raise CompileException(f"File not found: {file_path}")

        # replace #include directives with their content
        def replace_include(match):
            lib_path = ROOT_GLSL / match.group(1)
            if lib_path not in include:
                return scan_include(lib_path, idx+1)
            return ""

        return RE_INCLUDE.sub(replace_include, result)

    return scan_include(fname)

def import_dynamic() -> Tuple[str,...]:
    ret = []
    sort = 4000
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
            sort_order -= 2500

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
            "PASS": [x.strip() for x in meta.get('pass', "").split(",") if len(x) > 0],
            "OUTPUT": [x.strip() for x in meta.get('output', "").split(",") if len(x) > 0],
            "SORT": sort_order,
        })

        sort += 10
        ret.append((class_name, class_def,))
    return ret
