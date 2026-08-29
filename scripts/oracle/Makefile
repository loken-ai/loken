# Builds the GGUF block-format oracle against a llama.cpp checkout.
#
#   LLAMA_CPP=/path/to/llama.cpp make -C scripts/oracle
#
# Nothing from llama.cpp is copied into this repository: the path is supplied, one source file
# of theirs is compiled, and the result is a tool that prints reference vectors. A path that is
# true on one machine does not belong in a file everyone gets.

LLAMA_CPP ?= $(error set LLAMA_CPP to a llama.cpp checkout, e.g. LLAMA_CPP=~/src/llama.cpp make)
GGML      := $(LLAMA_CPP)/ggml

CFLAGS  := -O2 -std=c11 -I$(GGML)/include -I$(GGML)/src -I$(GGML)/src/ggml-cpu
LDFLAGS := -lm

gguf_quant_oracle: gguf_quant_oracle.c $(GGML)/src/ggml-quants.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# The vectors are checked in, so a contributor without llama.cpp can still run the parity
# tests. Regenerate only when the block formats themselves change.
vectors: gguf_quant_oracle
	./gguf_quant_oracle > vectors.txt
	@echo "wrote $$(wc -l < vectors.txt) lines"

clean:
	rm -f gguf_quant_oracle

.PHONY: vectors clean
