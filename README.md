<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Amorano/Jovimetrix-examples/blob/master/res/logo-jovimetrix-light.png">
  <img alt="ComfyUI Nodes for creating GLSL shaders">
</picture>

<h2><div align="center">
<a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI</a> Nodes for creating GLSL shaders
</div></h2>

<h3><div align="center">
JOVI_GLSL IS ONLY GUARANTEED TO SUPPORT <a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI 0.3.7+</a> and <a href="https://github.com/Comfy-Org/ComfyUI_frontend">FRONTEND 1.6.2+</a><br>
IF YOU NEED AN OLDER VERSION, PLEASE DO NOT UPDATE.
</div></h3>

<h2><div align="center">

![KNIVES!](https://badgen.net/github/open-issues/Amorano/JOVI_GLSL)
![FORKS!](https://badgen.net/github/forks/Amorano/JOVI_GLSL)

</div></h2>

<!---------------------------------------------------------------------------->

# SPONSORSHIP

Please consider sponsoring me if you enjoy the results of my work, code or documentation or otherwise. A good way to keep code development open and free is through sponsorship.

<div align="center">

[![BE A GITHUB SPONSOR ❤️](https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA)](https://github.com/sponsors/Amorano)

[![DIRECTLY SUPPORT ME VIA PAYPAL](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://www.paypal.com/paypalme/onarom)

[![PATREON SUPPORTER](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/joviex)

[![SUPPORT ME ON KO-FI!](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/alexandermorano)

</div>

## HIGHLIGHTS

* `GLSL Node`  provides raw access to user written Vertex and Fragment shaders to test at runtime
* `Dynamic GLSL` dynamically convert existing GLSL script files into ComfyUI nodes at load time
* Supports vector types for 2, 3 and 4 sized tuples (integer or float)
* Specific RGB/RGBA color vector support with access to the system/browser level color picker
* All `Image` inputs support RGB, RGBA or pure MASK input
* Over a dozen hand written GLSL nodes to speed up specific tasks better done on the GPU (10x speedup in most cases)

## UPDATES

**2024/12/19** @1.0.0:
* initial release

# INSTALLATION

[Please see the wiki for advanced use of the environment variables that can be used at startup](https://github.com/Amorano/Jovi_GLSL/wiki)

## COMFYUI MANAGER

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, simply search for Jovi_GLSL and install from the manager's database.

## MANUAL INSTALL
Clone the repository into your ComfyUI custom_nodes directory. You can clone the repository with the command:
```
git clone https://github.com/Amorano/Jovi_GLSL.git
```
You can then install the requirements by using the command:
```
.\python_embed\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\Jovi_GLSL\requirements.txt
```
If you are using a <code>virtual environment</code> (<code><i>venv</i></code>), make sure it is activated before installation. Then install the requirements with the command:
```
pip install -r .\ComfyUI\custom_nodes\Jovi_GLSL\requirements.txt
```
# WHERE TO FIND ME

You can find me on [![DISCORD](https://dcbadge.vercel.app/api/server/62TJaZ3Z5r?style=flat-square)](https://discord.gg/62TJaZ3Z5r).
