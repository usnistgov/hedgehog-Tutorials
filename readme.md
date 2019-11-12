# Hedgehog Tutorials

In this repository we present the tutorials for the [Hedgehog API](https://github.com/usnistgov/hedgehog).

## Content
- [Tutorials](#tutorials)
- [Dependencies](#dependencies)
- [Getting started](#getting-started)
- [Credits](#credits)
- [Contact Us](#contact-us)
- [Disclaimer](#disclaimer)

## Tutorials
Detailed explanations about the tutorials can be found in this [website](). 

## Dependencies
- All tutorials depends on the [Hedgehog](https://github.com/usnistgov/hedgehog) library, location is specified by using: 
``` cmake
cmake -DHedgehog_INCLUDE_DIR=<dir>
```
The tutorials presented here need to be compiled with a C++ compiler compatible with C++17 with the standard filesystem library accessible. 

Tested Compilers and Debuggers:
- Linux
  + g++ 8.3.0
  + GDB 8.3
- Windows: 
  + cl.exe 19.23 (MSVC 14.23.28105)
  + GDB 8.30
- macOS 10.15
  + g++ 8.3.0
  + GDB 8.0.1
  + LLDB 7.0.1
  
For a custom GDB, [CLion](https://www.jetbrains.com/clion/) supports version 7.8.x-8.3.x

Libraries/tools:
- cmake 3.12+ is recommended for building the tutorials.
- Tutorial 3 requires [OpenBlas](http://www.openblas.net/).
- Tutorial 4 requires [CUDA](https://developer.nvidia.com/cuda-zone).

[TCLAP](http://tclap.sourceforge.net/) is used and is embedded in the repository to parse the command-line of the
 different tutorials. 

## Getting started
```
1) From the hedgehog-Tutorials directory, create a build directory
2) From the build directory run, 'cmake -DHedgehog_INCLUDE_DIR=/path/to/hedgehog ../'
3) Run 'make'
```

The firsts tutorial are meant to demonstrate the usage of Hedgehog's API, and are not representative of gaining performance.


## Credits

Alexandre Bardakoff

Timothy Blattner

Walid Keyrouz

Bruno Bachelet

Loïc Yon

Mary Brady

## Contact Us

<a target="_blank" href="mailto:alexandre.bardakoff@nist.gov">Alexandre Bardakoff (alexandre.bardakoff ( at ) nist.gov)</a>

<a target="_blank" href="mailto:timothy.blattner@nist.gov">Timothy Blattner (timothy.blattner ( at ) nist.gov)</a>

## Disclaimer

NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

