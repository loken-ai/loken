// Device memory a launcher keeps for one stream and grows on demand.
//
// One buffer per use, stream and card.
// The kernels that read the buffer run on the stream that wrote it, and that stream orders them.
// A host lock does not order the card, so two streams never share a buffer.
//
// Several host threads may submit to one stream.
// A caller holds the reservation until every launch that reads the buffer is on the stream.
// Another thread's launches then land before or after them, never between.
//
// A larger request takes a larger buffer.
// The smaller one stays allocated, since a recorded graph may still read it.
// A refused growth keeps the slot's buffer and size, and the caller gets a null pointer.
//
// The buffers of a card live until `loken_stream_scratch_release` runs on it.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <mutex>

// What a buffer holds. Each use has a table of its own.
enum stream_scratch_use {
    STREAM_SCRATCH_Y_Q8_1,     // the activation quantised to q8_1
    STREAM_SCRATCH_ROUTING,    // the expert offsets and their histogram
    STREAM_SCRATCH_PAIR_ROWS,  // one output row per routed pair
    STREAM_SCRATCH_USES,
};

// The buffer, held for as long as the caller needs it. Null when the card refused the growth.
struct stream_scratch_reservation {
    std::unique_lock<std::mutex> held;
    void* ptr;
    operator void*() const { return ptr; }
};

// A buffer of at least `needed` bytes for `stream` on the current card.
stream_scratch_reservation reserve_stream_scratch(stream_scratch_use use, size_t needed,
                                                  cudaStream_t stream);

// Frees every buffer of the current card, of every use. Nothing on the card may still read them.
extern "C" void loken_stream_scratch_release();
