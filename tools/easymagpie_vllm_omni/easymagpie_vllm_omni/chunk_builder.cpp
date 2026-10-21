// CPU-only fixed-capacity codec chunk builder. One owner serializes calls.
// The ctypes CDLL entrypoint releases the Python GIL for the complete batch.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {
constexpr uint8_t HAS_FRAME = 1, SKIP_ZERO = 2, TERMINAL = 4, SEGMENT = 8, ABORT = 16;
constexpr uint8_t CHUNK = 1, EMPTY = 2, MARKER = 3, STALE = 4, ABORTED = 5;
struct Lane {
    uint64_t generation = 0, seen = 0, emitted = 0, chunks = 0;
    uint32_t delay = 0, pending = 0;
    bool active = false;
    int64_t rows[8 * 16]{};
};
struct Builder {
    Builder(uint32_t capacity, uint32_t count, const uint8_t *frames) : lanes(capacity) {
        if (count) startup.assign(frames, frames + count);
    }
    std::vector<Lane> lanes;
    // Owned, immutable after creation; reused generations restart their chunk counter.
    std::vector<uint8_t> startup;
};
}

extern "C" {
void *cb_create_configured(uint32_t capacity, uint32_t count, const uint8_t *startup) noexcept {
    if (!capacity || capacity > 1048576 || count > 4096 || (count && !startup)) return nullptr;
    for (uint32_t i = 0; i < count; ++i) if (!startup[i] || startup[i] > 8) return nullptr;
    try { return new Builder(capacity, count, startup); } catch (...) { return nullptr; }
}
void *cb_create(uint32_t capacity) noexcept {
    static constexpr uint8_t startup[] = {2, 6};
    return cb_create_configured(capacity, 2, startup);
}
void cb_destroy(void *opaque) noexcept { delete static_cast<Builder *>(opaque); }
// Generation must increase when reusing a slot. Zero is reserved.
int cb_reset(void *opaque, uint32_t slot, uint64_t generation, uint32_t delay) noexcept {
    if (!opaque) return -1;
    auto &lanes = static_cast<Builder *>(opaque)->lanes;
    if (slot >= lanes.size() || !generation) return -2;
    if (generation <= lanes[slot].generation) return -3;
    lanes[slot] = Lane{};
    lanes[slot].generation = generation;
    lanes[slot].delay = delay;
    lanes[slot].active = true;
    return 0;
}
int cb_release(void *opaque, uint32_t slot, uint64_t generation) noexcept {
    if (!opaque) return -1;
    auto &lanes = static_cast<Builder *>(opaque)->lanes;
    if (slot >= lanes.size()) return -2;
    if (lanes[slot].generation != generation) return 1;
    lanes[slot].active = false;
    lanes[slot].pending = 0;
    return 0;
}
int cb_release_batch(void *opaque, uint32_t count, const int32_t *slots,
                     const uint64_t *generations) noexcept {
    if (!opaque || (count && (!slots || !generations))) return -1;
    auto &lanes = static_cast<Builder *>(opaque)->lanes;
    for (uint32_t i = 0; i < count; ++i) {
        if (slots[i] < 0 || static_cast<uint32_t>(slots[i]) >= lanes.size()) return -2;
    }
    for (uint32_t i = 0; i < count; ++i) cb_release(opaque, slots[i], generations[i]);
    return 0;
}
int cb_reset_batch(void *opaque, uint32_t count, const int32_t *slots,
                   const uint64_t *generations, const uint32_t *delays) noexcept {
    if (!opaque || (count && (!slots || !generations || !delays))) return -1;
    auto &lanes = static_cast<Builder *>(opaque)->lanes;
    for (uint32_t i = 0; i < count; ++i) {
        if (slots[i] < 0 || static_cast<uint32_t>(slots[i]) >= lanes.size() ||
            !generations[i] || generations[i] <= lanes[slots[i]].generation) return -2;
        for (uint32_t j = 0; j < i; ++j) if (slots[i] == slots[j]) return -3;
    }
    for (uint32_t i = 0; i < count; ++i) cb_reset(opaque, slots[i], generations[i], delays[i]);
    return 0;
}
// Output indexed by input event. kind=0 means no publish; chunk rows are 8x16
// with unused rows set to -1. out_frames=1..8 for chunks, 0 for controls.
// Input/output arrays have count elements and rows have count*8*16 elements.
// Validate the entire input before changing state, so invalid inputs are atomic.
int cb_append(void *opaque, uint32_t count, const int64_t *rows,
              const int32_t *slots, const uint64_t *generations,
              const uint8_t *flags, int64_t *out_rows,
              uint8_t *out_frames, uint8_t *out_kind,
              uint8_t *out_finished, uint64_t *out_counters) noexcept {
    if (!opaque || (count && (!rows || !slots || !generations || !flags ||
                             !out_rows || !out_frames || !out_kind || !out_finished || !out_counters))) return -1;
    auto &lanes = static_cast<Builder *>(opaque)->lanes;
    for (uint32_t i = 0; i < count; ++i) {
        if (slots[i] < 0 || static_cast<uint32_t>(slots[i]) >= lanes.size()) return -2;
        if (flags[i] & ~uint8_t(31)) return -3;
    }
    std::memset(out_kind, 0, count);
    std::memset(out_frames, 0, count);
    std::memset(out_finished, 0, count);
    for (uint32_t i = 0; i < count; ++i) {
        auto &lane = lanes[slots[i]];
        const uint8_t f = flags[i];
        if (!lane.active || lane.generation != generations[i]) {
            out_kind[i] = STALE;
            continue;
        }
        if (f & ABORT) {
            lane.active = false;
            lane.pending = 0;
            out_kind[i] = ABORTED;
            continue;
        }
        bool has_frame = f & HAS_FRAME;
        const int64_t *row = rows + static_cast<size_t>(i) * 16;
        if (has_frame && (f & SKIP_ZERO)) {
            has_frame = false;
            for (int j = 0; j < 16; ++j) has_frame |= row[j] != 0;
        }
        if (has_frame && ++lane.seen > lane.delay) {
            std::memcpy(lane.rows + lane.pending * 16, row, 16 * sizeof(int64_t));
            ++lane.pending;
        }
        const bool terminal = f & TERMINAL;
        const bool finished = terminal || (f & SEGMENT);
        const auto &startup = static_cast<Builder *>(opaque)->startup;
        const uint32_t target = lane.chunks < startup.size() ? startup[lane.chunks] : 8;
        if (lane.pending && (terminal || lane.pending >= target)) {
            int64_t *output = out_rows + static_cast<size_t>(i) * 8 * 16;
            std::fill(output, output + 8 * 16, int64_t(-1));
            std::memcpy(output, lane.rows, lane.pending * 16 * sizeof(int64_t));
            out_frames[i] = static_cast<uint8_t>(lane.pending);
            out_kind[i] = CHUNK;
            out_finished[i] = finished;
            lane.emitted += lane.pending;
            ++lane.chunks;
            lane.pending = 0;
        } else if (finished) {
            out_kind[i] = (lane.pending == 0 && lane.emitted == 0) ? EMPTY : MARKER;
            out_finished[i] = true;
        }
        if (terminal) lane.active = false;
        out_counters[i * 3] = lane.seen;
        out_counters[i * 3 + 1] = lane.emitted;
        out_counters[i * 3 + 2] = lane.chunks;
    }
    return 0;
}
// metadata: generation,seen,emitted,chunks,delay,pending,active.
int cb_export(void *opaque, uint32_t slot, uint64_t *metadata, int64_t *rows) noexcept {
    if (!opaque || !metadata || !rows) return -1;
    auto &lanes = static_cast<Builder *>(opaque)->lanes;
    if (slot >= lanes.size()) return -2;
    const auto &lane = lanes[slot];
    metadata[0] = lane.generation; metadata[1] = lane.seen;
    metadata[2] = lane.emitted; metadata[3] = lane.chunks;
    metadata[4] = lane.delay; metadata[5] = lane.pending;
    metadata[6] = lane.active;
    std::memcpy(rows, lane.rows, sizeof(lane.rows));
    return 0;
}
}
