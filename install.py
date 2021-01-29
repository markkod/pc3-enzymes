import os


def install_dependencies():
    stream = os.popen('git clone https://github.com/chrsmrrs/tudataset.git && \
            pip --no-cache-dir install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html && \
            pip --no-cache-dir install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html  && \
            pip --no-cache-dir install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html  && \
            pip --no-cache-dir install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html  && \
            pip --no-cache-dir install torch-geometric && \
            pip --no-cache-dir install pybind11 && \
            sudo apt-get install libeigen3-dev && \
            cd .. && \
            cd /content/tudataset/tud_benchmark/kernel_baselines/ && \
            g++ -I /usr/include/eigen3 -03 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` kernel_baselines.cpp src/*cpp -o ../kernel_baselines`python3-config --extension-suffix`')
    output = stream.read()
    print(output)
