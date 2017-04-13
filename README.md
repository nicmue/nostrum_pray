# Nostrum Pray - Parallel Ray- and Pathtracer

## About

Nostrum Pray is a massive parallel ray- and pathtracer, developed in a lab for multicore development at Karlsruhe Institut für Technologie.

* Used data structure: KD Tree, built with OpenMP task parallelization on CPU
* Ray tracing: OpenMP-parallelized SIMD (AVX resp. SSE3) intersections on CPU
* Path tracing: naive recursive GPU path tracer with changed call stack size and transformed KD tree datastructure

Contributors:
* Manuel Karl
* Dominik Kleiser
* Marc Leinweber <post@mleinweber.de>
* Nico Muerdter <nicomuerdter@gmail.com>

## Compile

The provided CMakeLists.txt is configured to compile the project with the features provided by your system. So if your machine can't handle AVX its compiled with SSE. If your machine is also not able to handle SSE you just get the plain c++ implementation. Same behaviour is configured for GPU usage.

``` bash
mkdir out
cd out
cmake ..
make
```

## Usage

``` bash
./Pray object.json out.bmp
```

The input file needs to be a json as specified in the example folder.

## Todo

* FIX: SAH -> or not... seems to be slow for current implementation
* FIX: Big object benchmark (WTF 480s?!)
* Improvement: ram usage of gpu tracer for small GPU (Host 205, only 1GB [big bench needs about 1.6GB for kd tree alone!])
* Improvement: SIMD parallelization of pre-tracing calculations