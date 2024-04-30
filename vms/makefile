#!/usr/bin/bash

CXX=g++
CXXFLAGS  = -std=c++11 -fopenmp -O2  -fPIC -c
SWIGFLAGS = -c++ -python -shadow

# location of the Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
#PYTHON_INCLUDE = /p/software/jusuf/stages/2022/software/Python/3.9.6-GCCcore-11.2.0/include/python3.9

all:  _sl_sdde.so

help: # add some explanation
	@echo "make all     (default)"
	@echo "make clean   (remove all .o, .so, .pyc, and swig generated files)"
	@echo "make sl_sdde (compile Stuart-Landau SDDE model)"


sl_sdde: _sl_sdde.so

# -----------------------------------------------------------------------------
_sl_sdde.so: sl_sdde_wrap.o
	$(CXX) -shared -fopenmp sl_sdde_wrap.o -o _sl_sdde.so

sl_sdde_wrap.o : sl_sdde_wrap.cpp sl_sdde.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

sl_sdde_wrap.cpp : sl_sdde.i
	swig $(SWIGFLAGS) -o sl_sdde_wrap.cpp sl_sdde.i
# -----------------------------------------------------------------------------

.PHONEY: help clean all sl_sdde do

clean :
	rm -rf *.o  *.so *.pyc
	rm -rf sl_sdde_wrap.cpp sl_sdde.py
	rm -rf __pycache__
