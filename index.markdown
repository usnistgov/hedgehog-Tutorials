---
layout: home
---

In this repository we present the tutorials for the [Hedgehog API](https://github.com/usnistgov/hedgehog). 
The source code for the tutorials can be found [Here](https://github.com/usnistgov/hedgehog-Tutorials).

# Content
- [Dependencies](#dependencies)
- [Getting started](#getting-started)
- [Tutorial contents](#tutorial-contents)
- [Credits](#credits)
- [Contact Us](#contact-us)
- [Disclaimer](#disclaimer)

# Dependencies
- All tutorials depend on the [Hedgehog](https://github.com/usnistgov/hedgehog) library, location is specified using: 
{% highlight cmake %}
cmake -DHedgehog_INCLUDE_DIR=<dir>
{% endhighlight %}

- Tutorial 3 requires <a href="http://www.openblas.net/" rel="external">OpenBLAS </a>
- Tutorial 4 and 5 require <a href="https://developer.nvidia.com/cuda-zone" rel="external">CUDA </a>.

<a href="http://tclap.sourceforge.net/" rel="external">TCLAP</a> is used and is embedded in the repository to parse the command of the
 different tutorials.

# Getting started
The tutorials presented here need to be compiled with a C++ compiler that is compatible with C++20 and has the standard filesystem library accessible. 

Tested Compilers:
- g++ 11.1+
- clang 10
- MSVC 14.33

To use the Hedgehog API include the following header file:
```
#include <hedgehog/hedgehog.h>
```
# Tutorial contents
The tutorials are meant to demonstrate the usage of the Hedgehog API, and are not meant for obtaining performance.
- [Preamble](tutorials/preamble.html)
- [Tutorial 1 - Graph and Task: Simple Hadamard product](tutorials/tutorial1.html)
- [Tutorial 2 - Multiple inputs, State, State Manager: Hadamard product](tutorials/tutorial2.html)
- [Tutorial 3 - Performance and graph reuse](tutorials/tutorial3.html)
- [Tutorial 4 - Cycle resolution: CPU Matrix Multiplication](tutorials/tutorial4.html)
- [Tutorial 5 - GPU Computation and memory management: GPU Matrix Multiplication](tutorials/tutorial5.html)
- [Tutorial 6 - MultiGPU Computation and graph bunble: GPU Matrix Multiplication](tutorials/tutorial6.html)
- [Tutorial 7 - Compile-time analysis](tutorials/tutorial7.html)
- [Profiling tools in Hedgehog](tutorials/y_profiling.html)
- [Moving from Hedgehog v.2 to v.3](tutorials/z_hh2tohh3.html)

# Credits

Alexandre Bardakoff

Timothy Blattner

Walid Keyrouz

Bruno Bachelet

Loïc Yon

Mary Brady

# Contact Us

<a target="_blank" href="mailto:alexandre.bardakoff@nist.gov">Alexandre Bardakoff (alexandre.bardakoff ( at ) nist.gov)</a>

<a target="_blank" href="mailto:timothy.blattner@nist.gov">Timothy Blattner (timothy.blattner ( at ) nist.gov)</a>

# Disclaimer

NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and distributing the software, and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

