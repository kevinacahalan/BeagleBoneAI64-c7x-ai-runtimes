CC = aarch64-linux-gnu-gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I. -I../../runtime/dsp/include

SRCS = main.c util.c shutdown.c trace.c bench.c llama.c
OBJS = $(SRCS:.c=.o)
HDRS = c7x_ai.h ../../runtime/dsp/include/c7x_ai_proto.h ../../runtime/dsp/include/c7x_llama_fmt.h

.PHONY: all clean

all: c7x-ai

c7x-ai: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) -lm

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f c7x-ai $(OBJS)
