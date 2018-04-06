CC=gcc
TARGET=main
OBJECTS=photomosaic.o qdbmp.o timer.o
MODE=single
SRC=../data
MYSRC=$(SRC)/mydata2
OUTPUT=./output
PNUM=A
NUM=0
DEBUG=0

# may change to O3
CFLAGS=-std=c99 -O2 -Wall -fopenmp -mavx -DDEBUG=$(DEBUG)
LDFLAGS=-lm

.PHONY: clean stats cleanout check my_check my_check2 mydata clear_out

all: $(TARGET)

$(TARGET): $(OBJECTS)

run: $(TARGET)
	thorq --add --mode $(MODE) --name $(PNUM)_$(NUM) \
		./main $(SRC)/input$(NUM).bmp $(OUTPUT)/$(PNUM)_output$(NUM).bmp

myrun: $(TARGET)
	thorq --add --mode $(MODE) --name $(PNUM)_$(S)$(VER)_conv \
		./$(TARGET) $(MYSRC)/ex_$(S)x$(S)$(VER).bmp $(OUTPUT)/$(PNUM)_$(S)x$(S)$(VER).bmp

myrun2: $(TARGET)
	thorq --add --mode $(MODE) --name $(PNUM)_$(S1)x$(S2)_conv \
		./$(TARGET) $(MYSRC)/ex_$(S1)x$(S2).bmp $(OUTPUT)/$(PNUM)_$(S1)x$(S2).bmp

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

mydata:
	ls $(MYSRC)
