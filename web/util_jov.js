/**
 * File: util_jov.js
 * Project: Jovi_GLSL
 *
 */

import { app } from "../../scripts/app.js"

const _REGEX = /\d/;

const _MAP = {
    STRING: "📝",
    BOOLEAN: "🇴",
    INT: "🔟",
    FLOAT: "🛟",
    VEC2: "🇽🇾",
    COORD2D: "🇽🇾",
    VEC2INT: "🇽🇾",
    VEC3: "🇽🇾\u200c🇿",
    VEC3INT: "🇽🇾\u200c🇿",
    VEC4: "🇽🇾\u200c🇿\u200c🇼",
    VEC4INT: "🇽🇾\u200c🇿\u200c🇼",
    LIST: "🧾",
    DICT: "📖",
    IMAGE: "🖼️",
    MASK: "😷"
}

export const CONVERTED_TYPE = "converted-widget"

// return the internal mapping type name
export function widget_type_name(type) { return _MAP[type];}

export function widget_get_type(config) {
    // Special handling for COMBO so we restrict links based on the entries
    let type = config?.[0]
    let linkType = type
    if (type instanceof Array) {
      type = 'COMBO'
      linkType = linkType.join(',')
    }
    return { type, linkType }
}

export const widgetFind = (widgets, name) => widgets.find(w => w.name == name);

export const widgetFindOutput = (widgets, name) => {
    for (let i = 0; i < widgets.length; i++) {
        if (widgets[i].name == name) {
            return i;
        }
    }
}

export function widgetRemove(node, widgetOrSlot) {
    let index = 0;
    if (typeof widgetOrSlot == 'number') {
        index = widgetOrSlot;
    }
    else if (widgetOrSlot) {
        index = node.widgets.indexOf(widgetOrSlot);
    }
    if (index > -1) {
        const w = node.widgets[index];
        if (w.canvas) {
            w.canvas.remove()
        }
        if (w.inputEl) {
            w.inputEl.remove()
        }
        w.onRemoved?.()
        node.widgets.splice(index, 1);
    }
}

export function widgetRemoveAll(node) {
    if (node.widgets) {
        for (const w of node.widgets) {
            widgetRemove(node, w);
        }
        node.widgets.length = 0;
    }
}

export function widgetHide(node, widget, suffix = '') {
    if ((widget?.hidden || false) || widget.type?.startsWith(CONVERTED_TYPE + suffix)) {
        return;
    }
    widget.origType = widget.type;
    widget.type = CONVERTED_TYPE + suffix;
    widget.hidden = true;

    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];

    widget.origSerializeValue = widget.serializeValue;
    widget.serializeValue = async () => {
        // Prevent serializing the widget if we have no input linked
        if (!node.inputs) {
            return undefined;
        }

        let node_input = node.inputs.find((i) => i.widget?.name == widget.name);
        if (!node_input || !node_input.link) {
            return undefined;
        }
        return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
    }

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            widgetHide(node, w, ':' + widget.name);
        }
    }
}

export function widgetShow(widget) {
    if (widget?.origType) {
        widget.type = widget.origType;
        delete widget.origType;
    }

    widget.computeSize = widget.origComputeSize;
    delete widget.origComputeSize;

    if (widget.origSerializeValue) {
        widget.serializeValue = widget.origSerializeValue;
        delete widget.origSerializeValue;
    }

    widget.hidden = false;
    if (widget?.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            widgetShow(w)
        }
    }
}

export function widgetShowVector(widget, values={}, type) {
    widgetShow(widget);
    if (["FLOAT"].includes(type)) {
        type = "VEC1";
    } else if (["INT"].includes(type)) {
        type = "VEC1INT";
    } else if (type == "BOOLEAN") {
        type = "toggle";
    }

    if (type !== undefined) {
        widget.type = type;
    }

    if (widget.value === undefined) {
        widget.value = widget.options?.default || {};
    }

    // convert widget.value to pure dict/object
    if (Array.isArray(widget.value)) {
        let new_val = {};
        for (let i = 0; i < widget.value.length; i++) {
            new_val[i] = widget.value[i];
        }
        widget.value = new_val;
    }

    widget.options.step = 1;
    widget.options.round = 1;
    widget.options.precision = 6;
    if (widget.type != 'toggle') {
        let size = 1;
        const match = _REGEX.exec(widget.type);
        if (match) {
            size = match[0];
        }
        if (!widget.type.endsWith('INT') && widget.type != 'BOOLEAN') {
            widget.options.step = 0.01;
            widget.options.round = 0.001;
        }

        widget.value = {};
        for (let i = 0; i < size; i++) {
            widget.value[i] = widget.type.endsWith('INT') ? Math.round(values[i]) : Number(values[i]);
        }
    } else {
        widget.value = values[0] ? true : false;
    }
}

export function widgetProcessAny(widget, subtype="FLOAT") {
    widgetShow(widget);
    //input.type = subtype;
    if (subtype == "BOOLEAN") {
        widget.type = "toggle";
    } else if (subtype == "FLOAT" || subtype == "INT") {
        widget.type = "number";
        if (widget?.options) {
            if (subtype=="FLOAT") {
                widget.options.precision = 3;
                widget.options.step = 1;
                widget.options.round = 0.1;
            } else {
                widget.options.precision = 0;
                widget.options.step = 10;
                widget.options.round = 1;
            }
        }
    } else {
        widget.type = subtype;
    }
}

export function widgetToWidget(node, widget) {
    widgetShow(widget);
    //const sz = node.size;
    node.removeInput(node.inputs.findIndex((i) => i.widget?.name == widget.name));

    for (const widget of node.widgets) {
        widget.last_y -= LiteGraph.NODE_SLOT_HEIGHT;
    }
    nodeFitHeight(node);

    // Restore original size but grow if needed
    //node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
    //node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}

export function widgetToInput(node, widget, config) {
    widgetHide(node, widget, '-jov');

    const { linkType } = widget_get_type(config);

    // Add input and store widget config for creating on primitive node
    //const sz = node.size
    node.addInput(widget.name, linkType, {
        widget: { name: widget.name, config },
    })

    for (const widget of node.widgets) {
        widget.last_y += LiteGraph.NODE_SLOT_HEIGHT;
    }
    nodeFitHeight(node);

    // Restore original size but grow if needed
    //node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function widgetGetHovered() {
    if (typeof app == 'undefined') return;

    const node = app.canvas.node_over;
    if (!node || !node.widgets) return;

    const graphPos = app.canvas.graph_mouse;
    const x = graphPos[0] - node.pos[0];
    const y = graphPos[1] - node.pos[1];

    let pos_y;
    for (const w of node.widgets) {
        let widgetWidth, widgetHeight;
        if (w.computeSize) {
            const sz = w.computeSize();
            widgetWidth = sz[0] || 0;
            widgetHeight = sz[1] || 0;
        } else {
            widgetWidth = w.width || node.size[0] || 0;
            widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT;
        }

        if (pos_y === undefined) {
            pos_y = w.last_y || 0;
        }

        if (widgetHeight > 0 && widgetWidth > 0 && w.last_y !== undefined && x >= 6 && x <= widgetWidth - 12 && y >= w.last_y && y <= w.last_y + widgetHeight) {
            return {
                widget: w,
                x1: 6 + node.pos[0],
                y1: node.pos[1] + w.last_y,
                x2: node.pos[0] + widgetWidth - 12,
                y2: node.pos[1] + w.last_y + widgetHeight
            };
        }
    }
}

export function nodeFitHeight(node) {
    const size_old = node.size;
    node.computeSize();
    node.setSize([Math.max(size_old[0], node.size[0]), Math.min(size_old[1], node.size[1])]);
    node.setDirtyCanvas(!0, !1);
    app.graph.setDirtyCanvas(!0, !1);
}

// flash status for each element
const flashStatusMap = new Map();

export async function flashBackgroundColor(element, duration, flashCount, color="red") {
    if (flashStatusMap.get(element)) {
        return;
    }

    flashStatusMap.set(element, true);
    const originalColor = element.style.backgroundColor;

    for (let i = 0; i < flashCount; i++) {
        element.style.backgroundColor = color;
        await new Promise(resolve => setTimeout(resolve, duration / 2));
        element.style.backgroundColor = originalColor;
        await new Promise(resolve => setTimeout(resolve, duration / 2));
    }
    flashStatusMap.set(element, false);
}

export function widgetSizeModeHook(nodeType, always_wh=false) {
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        const wh = this.widgets.find(w => w.name == 'WH');
        const samp = this.widgets.find(w => w.name == 'SAMPLE');
        const mode = this.widgets.find(w => w.name == 'MODE');
        mode.callback = () => {
            widgetHide(this, wh);
            widgetHide(this, samp);

            if (always_wh || !['MATTE'].includes(mode.value)) {
                widgetShow(wh);
            }
            if (!['CROP', 'MATTE'].includes(mode.value)) {
                widgetShow(samp);
            }
            nodeFitHeight(this);
        }
        setTimeout(() => { mode.callback(); }, 20);
        return me;
    }
}

export function widgetOutputHookType(node, control_key, match_output=0) {
    const combo = node.widgets.find(w => w.name == control_key);
    const output = node.outputs[match_output];

    if (!output || !combo) {
        throw new Error("Required widgets not found");
    }

    const oldCallback = combo.callback;
    combo.callback = () => {
        const me = oldCallback?.apply(this, arguments);
        node.outputs[match_output].name = widget_type_name(combo.value);
        node.outputs[match_output].type = combo.value;
        return me;
    }
    setTimeout(() => { combo.callback(); }, 10);
}

/*
* matchFloatSize forces the target to be float[n] based on its type size
*/
export function widgetHookAB(node, control_key, output_type_match=true) {

    const AA = node.widgets.find(w => w.name == '🅰️🅰️');
    const BB = node.widgets.find(w => w.name == '🅱️🅱️');
    const combo = node.widgets.find(w => w.name == control_key);

    if (combo === undefined) {
        return;
    }

    widgetHookControl(node, control_key, AA);
    widgetHookControl(node, control_key, BB);
    if (output_type_match) {
        widgetOutputHookType(node, control_key);
    }
    setTimeout(() => { combo.callback(); }, 5);

    return combo;
};

/*
* matchFloatSize forces the target to be float[n] based on its type size
*/
export function widgetHookControl(node, control_key, target, matchFloatSize=false) {
    const initializeTrack = (widget) => {
        const track = {};
        for (let i = 0; i < 4; i++) {
            track[i] = widget.options?.default[i];
        }
        Object.assign(track, widget.value);
        return track;
    };

    const { widgets } = node;
    const combo = widgets.find(w => w.name == control_key);

    if (!target || !combo) {
        throw new Error("Required widgets not found");
    }

    const data = {
        //track_xyzw: target.options?.default, //initializeTrack(target),
        track_xyzw: initializeTrack(target),
        target,
        combo
    };

    const oldCallback = combo.callback;
    combo.callback = () => {
        const me = oldCallback?.apply(this, arguments);
        widgetHide(node, target, "-jov");
        if (["VEC2", "VEC2INT", "COORD2D", "VEC3", "VEC3INT", "VEC4", "VEC4INT", "BOOLEAN", "INT", "FLOAT"].includes(combo.value)) {
            let type = combo.value;
            if (matchFloatSize) {
                type = "FLOAT";
                if (["VEC2", "VEC2INT", "COORD2D"].includes(combo.value)) {
                    type = "VEC2";
                } else if (["VEC3", "VEC3INT"].includes(combo.value)) {
                    type = "VEC3";
                } else if (["VEC4", "VEC4INT"].includes(combo.value)) {
                    type = "VEC4";
                }
            }
            widgetShowVector(target, data.track_xyzw, type);
        }
        nodeFitHeight(node);
        return me;
    }

    target.options.menu = false;
    target.callback = () => {
        if (target.type == "toggle") {
            data.track_xyzw[0] = target.value ? 1 : 0;
        } else {
            Object.keys(target.value).forEach((key) => {
                data.track_xyzw[key] = target.value[key];
            });
        }
    };

    return data;
}


function arrayToObject(values, length, parseFn) {
    const result = {};
    for (let i = 0; i < length; i++) {
        result[i] = parseFn(values[i]);
    }
    return result;
}

export function domRenderTemplate(template, data) {
    for (const key in data) {
        if (Object.prototype.hasOwnProperty.call(data, key)) {
            const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g')
            template = template.replace(regex, data[key])
        }
    }
    return template
}

export function domFoldableToggle(elementId, symbolId) {
    const content = document.getElementById(elementId)
    const symbol = document.getElementById(symbolId)
    if (content.style.display == 'none' || content.style.display == '') {
        content.style.display = 'flex'
        symbol.innerHTML = '&#9661' // Down arrow
    } else {
        content.style.display = 'none'
        symbol.innerHTML = '&#9655' // Right arrow
    }
}

export function domInnerValueChange(node, pos, widget, value, event=undefined) {
    const type = widget.type.includes("INT") ? Number : parseFloat
    widget.value = arrayToObject(value, Object.keys(value).length, type);
    if (
        widget.options &&
        widget.options.property &&
        node.properties[widget.options.property] !== undefined
        ) {
            node.setProperty(widget.options.property, widget.value)
        }
    if (widget.callback) {

        widget.callback(widget.value, app.canvas, node, pos, event)
    }
}

export function domWidgetOffset(
    widget,
    ctx,
    node,
    widgetWidth,
    widgetY,
    height
  ) {
    const margin = 10
    const elRect = ctx.canvas.getBoundingClientRect()
    const transform = new DOMMatrix()
      .scaleSelf(
        elRect.width / ctx.canvas.width,
        elRect.height / ctx.canvas.height
      )
      .multiplySelf(ctx.getTransform())
      .translateSelf(0, widgetY + margin)

    const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
    Object.assign(widget.inputEl.style, {
      transformOrigin: '0 0',
      transform: scale,
      left: `${transform.e}px`,
      top: `${transform.d + transform.f}px`,
      width: `${widgetWidth}px`,
      height: `${(height || widget.parent?.inputHeight || 32) - margin}px`,
      position: 'absolute',
      background: !node.color ? '' : node.color,
      color: !node.color ? '' : 'white',
      zIndex: 5, //app.graph._nodes.indexOf(node),
    })
}

export function domEscapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;')
}

export function domShowModal(innerHTML, eventCallback, timeout=null) {
    return new Promise((resolve, reject) => {
        const modal = document.createElement("div");
        modal.className = "modal";
        modal.innerHTML = innerHTML;
        document.body.appendChild(modal);

        // center
        const modalContent = modal.querySelector(".jov-modal-content");
        modalContent.style.position = "absolute";
        modalContent.style.left = "50%";
        modalContent.style.top = "50%";
        modalContent.style.transform = "translate(-50%, -50%)";

        let timeoutId;

        const handleEvent = (event) => {
            const targetId = event.target.id;
            const result = eventCallback(targetId);

            if (result != null) {
                if (timeoutId) {
                    clearTimeout(timeoutId);
                    timeoutId = null;
                }
                modal.remove();
                resolve(result);
            }
        };
        modalContent.addEventListener("click", handleEvent);
        modalContent.addEventListener("dblclick", handleEvent);

        if (timeout) {
            timeout *= 1000;
            timeoutId = setTimeout(() => {
                modal.remove();
                reject(new Error("TIMEOUT"));
            }, timeout);
        }

        //setTimeout(() => {
        //    modal.dispatchEvent(new Event('tick'));
        //}, 1000);
    });
}

export function colorHex2RGB(hex) {
  hex = hex.replace(/^#/, '');
  const bigint = parseInt(hex, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return [r, g, b];
}

/*
* Parse a string "255,255,255,255" or a List[255,255,255,255] into hex
*/
export function colorRGB2Hex(input) {
    const rgbArray = typeof input == 'string' ? input.match(/\d+/g) : input;
    if (rgbArray.length < 3) {
        throw new Error('input not 3 or 4 values');
    }
    const hexValues = rgbArray.map((value, index) => {
        if (index == 3 && !value) return 'ff';
        const hex = parseInt(value).toString(16);
        return hex.length == 1 ? '0' + hex : hex;
    });
    return '#' + hexValues.slice(0, 3).join('') + (hexValues[3] || '');
}

export function colorLerpHex(colorStart, colorEnd, lerp) {
  // Parse color strings into RGB arrays
  const startRGB = colorHex2RGB(colorStart);
  const endRGB = colorHex2RGB(colorEnd);

  // Linearly interpolate each RGB component
  const lerpedRGB = startRGB.map((component, index) => {
      return Math.round(component + (endRGB[index] - component) * lerp);
  });

  // Convert the interpolated RGB values back to a hex color string
  return colorRGB2Hex(lerpedRGB);
}

export function colorContrast(hexColor) {
    const rgb = colorHex2RGB(hexColor);
    const L = 0.2126 * rgb[0] / 255. + 0.7152 * rgb[1] / 255. + 0.0722 * rgb[2] / 255.;
    return L > 0.790 ? "#000" : "#CCC";
}
