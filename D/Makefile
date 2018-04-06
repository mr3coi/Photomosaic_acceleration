# Settings
CC=mpicc
TARGET=main
MODE=mpi
OBJECTS=photomosaic.o qdbmp.o timer.o helper_func.o
CFLAGS=-std=c99 -O2 -Wall -fopenmp -DDEBUG=$(DEBUG) -DPROF=$(PROF)
LDFLAGS=-lm -lOpenCL

# Directories
SRC=../data
MYSRC=$(SRC)/mydata2
OUTPUT=./output

# Constants (Initial values)
NUM=0
PNUM=D
DEBUG=0
PROF=0
NODE=4
SLOT=2

.PHONY: clean stats cleanout check clena my_check my_check2 mydata

all: $(TARGET)

$(TARGET): $(OBJECTS)

run: $(TARGET)
	thorq --add --mode $(MODE) --node $(NODE) --slot $(SLOT) --device gpu/7970 --name $(PNUM)_$(NUM) ./$(TARGET) $(SRC)/input$(NUM).bmp $(OUTPUT)/$(PNUM)_output$(NUM).bmp

myrun: $(TARGET)
	thorq --add --mode $(MODE) --node $(NODE) --slot $(SLOT) --device gpu/7970 --name $(PNUM)_$(S)$(VER)_conv ./$(TARGET) $(MYSRC)/ex_$(S)x$(S)$(VER).bmp $(OUTPUT)/$(PNUM)_$(S)x$(S)$(VER).bmp

myrun2: $(TARGET)
	thorq --add --mode $(MODE) --node $(NODE) --slot $(SLOT) --device gpu/7970 --name $(PNUM)_$(S1)x$(S2)_conv ./$(TARGET) $(MYSRC)/ex_$(S1)x$(S2).bmp $(OUTPUT)/$(PNUM)_$(S1)x$(S2).bmp

clean:
	rm -rf $(TARGET) $(OBJECTS)

stats:
	thorq --stat-all

cleanout:
	rm -f *.std*

check:
	diff $(OUTPUT)/$(PNUM)_output$(NUM).bmp $(SRC)/output$(NUM).bmp -s

my_check:
	diff $(OUTPUT)/$(PNUM)_$(S)x$(S)$(VER).bmp $(MYSRC)/output/out_$(S)x$(S)$(VER).bmp -s

my_check2:
	diff $(OUTPUT)/$(PNUM)_$(S1)x$(S2).bmp $(MYSRC)/output/out_$(S1)x$(S2).bmp -s

clena: clean

run_p: $(TARGET)
	PROF=1
	thorq --add --mode $(MODE) --device gpu/7970 --node $(NODE) --slot $(SLOT) --name $(PNUM)_$(NUM) /opt/CodeXL/CodeXLGpuProfiler -tpO ./$(TARGET) $(SRC)/input$(NUM).bmp $(OUTPUT)/$(PNUM)_output$(NUM).bmp

myrun_p: $(TARGET)
	thorq --add --mode $(MODE) --device gpu/7970 --node $(NODE) --slot $(SLOT) --name $(PNUM)_$(S)$(VER)_conv /opt/CodeXL/CodeXLGpuProfiler -tpO ./$(TARGET) $(MYSRC)/ex_$(S)x$(S)$(VER).bmp $(OUTPUT)/$(PNUM)_$(S)x$(S)$(VER).bmp

myrun2_p: $(TARGET)
	thorq --add --mode $(MODE) --node $(NODE) --slot $(SLOT) --device gpu/7970 --name $(PNUM)_$(S1)x$(S2)_conv /opt/CodeXL/CodeXLGpuProfiler -tpO ./$(TARGET) $(MYSRC)/ex_$(S1)x$(S2).bmp $(OUTPUT)/$(PNUM)_$(S1)x$(S2).bmp

mydata:
	ls $(MYSRC)
