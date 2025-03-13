CC = nvcc
CFLAGS = -g -I$(CUDA_HOME)/include  # Removed -Wall
SRC = deviceProperties.c
TARGET = deviceProperties

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

all: $(TARGET)