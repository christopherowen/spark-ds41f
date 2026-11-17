/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Disk readers for DeepSeek-V4.1's Engram tables on one rank.
 *
 * One worker thread serves every reader's steps in the order the model submits
 * them: per step, the first Engram layer's table, then the second's, so the rows
 * the forward needs first are read first and the two tables never compete for the
 * disk. A step's lookups split by when their row IDs exist. Those the host already
 * hashed (prompt tokens: the runner prepares a step before the GPU runs it) come
 * with the step as its early IDs, and the worker reads them as soon as the GPU has
 * finished gathering the reader's previous step (the GPU writes that step's
 * sequence number into the control block after its gather), so they are read while
 * the previous step computes. The rest (tokens the GPU samples) are known when the
 * GPU has written the step's sequence number into the control block (the stream
 * writes it after copying the row IDs to the host); the worker then reads those,
 * and publishes the step by writing its sequence number into the control block. A
 * step with no early IDs is read entirely the second way; one whose IDs are all
 * early does not wait for the GPU's. The worker never takes the GIL and never calls CUDA, so nothing
 * another thread does in Python or in the CUDA driver can hold back the rows. A
 * failed step is recorded and not published, and the control block says so: the
 * GPU's wait then stops it at once with a named record instead of letting it read
 * stale rows.
 *
 * Request planning (_row_plan.h) and io_uring execution come from b12x's PLE
 * reader (local-inference-lab/b12x b12x/loader/_ple_reader.c, Apache-2.0): rows
 * are sorted, deduplicated and coalesced into 4 KiB-aligned O_DIRECT reads into
 * registered staging buffers, then scattered to their lookup slots. A short read
 * continues into the rest of its buffer; only the end of the file fails it.
 */
#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char message[512];
} failure_t;

static void system_error(failure_t *failure, const char *operation) {
    snprintf(failure->message, sizeof(failure->message), "%s: %s", operation, strerror(errno));
}

#include "_row_plan.h"

#define READER_CAPSULE "vllm.engram_rows.reader"
#define ABI_VERSION 5

/* Control block: 32-bit words in mapped host memory, one 64-byte line per writer. */
#define CTRL_IDS_SEQ 0      /* the GPU stream: the step whose row IDs are on the host */
#define CTRL_CONSUMED_SEQ 1 /* the GPU: the step whose rows it has gathered */
#define CTRL_ROWS_SEQ 16    /* the worker: the step whose rows are in the buffers */
#define CTRL_TIMEOUT_US 32  /* the host: how long the GPU waits for a step's rows */
#define CTRL_WAITED_US 33   /* the GPU: how long a timed-out wait waited */
#define CTRL_FAILED_SEQ 34  /* the GPU: the step a timed-out wait was for */
#define CTRL_GPU_WAIT_SEQ 40 /* debug (timeline): the step the GPU last waited for */
#define CTRL_GPU_WAIT_NS 41  /* debug (timeline): how long that wait took, ns */
#define CTRL_PHASE 48       /* the worker: PHASE_* */
#define CTRL_PHASE_SEQ 49   /* the worker: the step it is on */
#define CTRL_WORDS 64

enum { PHASE_IDLE = 0, PHASE_WAITING_IDS = 1, PHASE_READING = 2, PHASE_FAILED = 3, PHASE_WAITING_CONSUMED = 4 };

/* An early ID the host does not know: the GPU's ID for that lookup is read later. */
#define UNKNOWN_ID INT64_MIN

#define QUEUE 64u /* steps queued across all readers */

typedef struct {
    char *buffer;
    size_t job;
    unsigned done; /* bytes of the job read so far: a short read continues */
} slot_t;

typedef struct {
    uint32_t sequence;
    uint32_t count;
    int64_t *early;   /* the host's IDs (UNKNOWN_ID where only the GPU knows), or NULL */
    bool has_unknown; /* some lookup needs the GPU's ID */
    bool is_prefetch; /* warm the user-space row cache without publishing a GPU step */
    uint64_t submitted; /* when the host queued the step (CLOCK_MONOTONIC ns) */
} step_t;

/* Debug (timeline): one record per served step, times in CLOCK_MONOTONIC ns. The GPU's
 * wait is the last one it recorded when the step's service began: the previous step's. */
typedef struct {
    uint32_t sequence, count, early, gpu_sequence, gpu_wait_ns;
    uint64_t submitted, started, consumed, ids, read, published, plan_ns;
} record_t;
#define TIMELINE 8192

typedef struct {
    ple_plan_t plan;
    bool poisoned;
    unsigned slots_count;
    slot_t *slots;
    char *buffers;
    struct io_uring ring;
    bool ring_ready;
    bool files_registered;
    int *registered_fds;
    struct iovec *iovecs;
    unsigned *free_slots;
    /* Mapped host buffers, owned by the Python wrapper for the reader's lifetime. */
    const char *ids;
    char *weights, *scales;
    char *early_weights, *early_scales;
    int64_t *prefetched_early;
    uint32_t prefetched_count;
    char *active_weights, *active_scales;
    uint32_t *control;
    int64_t *late; /* the GPU's IDs of a step's lookups the host did not know, -1 elsewhere */
    bool check_early; /* debug: an early ID must equal the GPU's */
    bool drop_page_cache; /* evict any VFS page-cache pages for read ranges */
    record_t *timeline; /* debug: the last TIMELINE served steps (mutex), or NULL */
    uint64_t timeline_next, timeline_read;
    uint64_t step_plan_ns; /* debug (timeline): the served step's planning time, worker only */
    /* Serving: started (sources fixed) and closing are set once; pending (steps
     * queued or being read) is guarded by the queue's mutex, the rest by mutex. */
    bool started, closing;
    unsigned pending;
    pthread_mutex_t mutex;
    uint64_t ids_timeout_ns, delay_ns;
    failure_t failure; /* the first failure; no step is read after it */
    uint64_t steps_done, lookups_done, early_lookups, fadvise_bytes;
    uint64_t early_prefetched_steps, prefetch_steps;
    uint64_t total_read_calls, total_unique_blocks, total_read_bytes;
    double ids_wait_seconds, read_seconds, consumed_wait_seconds, early_read_seconds;
} reader_t;

static uint64_t now_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static bool reached(uint32_t value, uint32_t target) { return (int32_t)(value - target) >= 0; }

static void control_store(reader_t *reader, int word, uint32_t value) {
    __atomic_store_n(&reader->control[word], value, __ATOMIC_RELAXED);
}

/* --- io_uring execution, from b12x's PLE reader ------------------------------ */

static void ple_error(reader_t *reader, const char *operation, int error) {
    if (!reader->plan.failure.message[0])
        snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message), "%s: %s",
                 operation, strerror(error));
}

/* Copy a completed job's fragments from its slot buffer to their lookup slots. */
static void ple_scatter(reader_t *reader, size_t index, const char *buffer) {
    ple_job_t *job = &reader->plan.jobs[index];
    char *dst_weights = reader->active_weights;
    char *dst_scales = reader->active_scales;
    if (!reader->plan.packed) {
        for (size_t i = job->begin; i < job->end; i++) {
            ple_fragment_t *fragment = &reader->plan.fragments[i];
            char *dst = (fragment->scale
                             ? dst_scales + (size_t)fragment->lookup * 8u
                             : dst_weights + (size_t)fragment->lookup * 256u) +
                        fragment->in_row;
            memcpy(dst, buffer + (fragment->offset - job->offset), fragment->length);
        }
    } else {
        for (size_t i = job->begin; i < job->end; i++) {
            ple_fragment_t *fragment = &reader->plan.fragments[i];
            ple_scatter_packed_fragment(fragment, buffer + (fragment->offset - job->offset),
                                        dst_weights, dst_scales);
        }
    }
    job->saved_buffer = buffer;
}

/* Queue a read of the rest of the slot's job into the rest of its buffer. */
static bool ple_prepare(reader_t *reader, unsigned slot) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&reader->ring);
    if (!sqe) {
        ple_error(reader, "io_uring SQ capacity", ENOSPC);
        reader->poisoned = true;
        return false;
    }
    slot_t *s = &reader->slots[slot];
    ple_job_t *job = &reader->plan.jobs[s->job];
    io_uring_prep_read_fixed(sqe, (int)job->file, s->buffer + s->done, job->length - s->done,
                             job->offset + s->done, (int)slot);
    sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, slot);
    return true;
}

/* Submit every prepared SQE. A short positive submit accounts for exactly that many
 * SQEs; retry only the rest. A failed submit may leave prepared SQEs, so stop
 * issuing, drain every submitted CQE, and retire the ring without submitting them. */
static void ple_flush(reader_t *reader, unsigned prepared, unsigned *outstanding) {
    while (prepared) {
        reader->plan.submit_calls++;
        int result = io_uring_submit(&reader->ring);
        if (result == -EINTR) continue;
        if (result <= 0) {
            ple_error(reader, "io_uring submit", result < 0 ? -result : EIO);
            reader->poisoned = true;
            return;
        }
        *outstanding += (unsigned)result;
        prepared -= (unsigned)result;
    }
}

/* Account one completion of the slot's job. Returns whether the slot is free again;
 * a partial read (or EAGAIN/EINTR) queues the rest and *continued counts it. */
static bool ple_complete(reader_t *reader, unsigned slot, int status, unsigned *continued) {
    slot_t *s = &reader->slots[slot];
    ple_job_t *job = &reader->plan.jobs[s->job];
    reader->plan.read_calls++;
    if (status == -EAGAIN || status == -EINTR) {
        if (ple_prepare(reader, slot)) {
            (*continued)++;
            return false;
        }
        return true;
    }
    if (status < 0) {
        ple_error(reader, "io_uring READ_FIXED", -status);
        return true;
    }
    reader->plan.read_bytes += (unsigned)status;
    s->done += (unsigned)status;
    if (s->done >= job->expected) {
        ple_scatter(reader, s->job, s->buffer);
        return true;
    }
    if (status == 0) {
        if (!reader->plan.failure.message[0])
            snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message),
                     "short Engram row read at offset %lld: the file ended %u bytes early; the "
                     "checkpoint changed or was truncated",
                     (long long)job->offset, job->expected - s->done);
        return true;
    }
    if (ple_prepare(reader, slot)) {
        (*continued)++;
        return false;
    }
    return true;
}

/* Assign the next jobs to free slots and queue their reads; returns how many. */
static unsigned ple_prepare_wave(reader_t *reader, unsigned free_count, size_t *next) {
    size_t remaining = reader->plan.job_count - *next;
    unsigned count = remaining < free_count ? (unsigned)remaining : free_count;
    unsigned prepared = 0;
    bool stash = *next > 0 && reader->plan.miss_count > 0 && !reader->plan.miss_src_weights;
    for (unsigned i = 0; i < count; i++) {
        unsigned slot = reader->free_slots[i];
        if (stash) ple_stash_job(&reader->plan, reader->slots[slot].job);
        reader->slots[slot].job = (*next)++;
        reader->slots[slot].done = 0;
        if (!ple_prepare(reader, slot)) break;
        prepared++;
    }
    return prepared;
}

static void ple_uring_run(reader_t *reader) {
    unsigned outstanding = 0;
    size_t next = 0;
    for (unsigned i = 0; i < reader->slots_count; i++) reader->free_slots[i] = i;
    ple_flush(reader, ple_prepare_wave(reader, reader->slots_count, &next), &outstanding);
    while (outstanding) {
        struct io_uring_cqe *cqe = NULL;
        /* wait/peek do not submit pending SQEs when draining a failed submit. */
        int result = io_uring_wait_cqe(&reader->ring, &cqe);
        if (result < 0) {
            if (result != -EINTR && result != -EAGAIN) {
                ple_error(reader, "io_uring completion wait", -result);
                reader->poisoned = true;
            }
            continue;
        }
        unsigned seen = 0, free_count = 0, continued = 0, head;
        io_uring_for_each_cqe(&reader->ring, head, cqe) {
            unsigned slot = (unsigned)io_uring_cqe_get_data64(cqe);
            int status = cqe->res;
            seen++;
            outstanding--;
            if (ple_complete(reader, slot, status, &continued)) reader->free_slots[free_count++] = slot;
            if (!outstanding) break;
        }
        io_uring_cq_advance(&reader->ring, seen);
        /* Do not prepare refill SQEs until every collected completion has been
         * checked. An I/O error therefore leaves no speculative refill entries. */
        unsigned prepared = continued;
        if (!reader->plan.failure.message[0] && !reader->poisoned && next < reader->plan.job_count)
            prepared += ple_prepare_wave(reader, free_count, &next);
        ple_flush(reader, prepared, &outstanding);
    }
    if (reader->poisoned) {
        io_uring_queue_exit(&reader->ring);
        reader->ring_ready = false;
    }
}

static int ple_register_files(ple_plan_t *plan, void *owner) {
    (void)plan;
    reader_t *reader = owner;
    int *fds = realloc(reader->registered_fds, reader->plan.file_count * sizeof(*fds));
    if (!fds) return -ENOMEM;
    reader->registered_fds = fds;
    for (size_t i = 0; i < reader->plan.file_count; i++) fds[i] = reader->plan.files[i].fd;
    if (reader->files_registered) {
        int error = io_uring_unregister_files(&reader->ring);
        if (error < 0) {
            reader->poisoned = true;
            return error;
        }
        reader->files_registered = false;
    }
    int error = io_uring_register_files(&reader->ring, fds, (unsigned)reader->plan.file_count);
    if (!error) reader->files_registered = true;
    else reader->poisoned = true;
    return error;
}

/* Read the rows of ``count`` lookups whose IDs are ``ids`` into ``dst_weights`` and
 * ``dst_scales``; an ID outside this rank's shard (or -1, or UNKNOWN_ID) leaves its slot
 * untouched. */
static failure_t read_rows_into(reader_t *reader, const char *ids, uint32_t count,
                                char *dst_weights, char *dst_scales, bool dst_is_cached) {
    reader->plan.failure.message[0] = 0;
    reader->plan.lookups = count;
    reader->plan.requested_bytes = reader->plan.read_bytes = reader->plan.read_calls = 0;
    reader->plan.unique_blocks = reader->plan.coalesced_reads = reader->plan.submit_calls = 0;
    if (reader->poisoned) {
        snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message),
                 "Engram reader is unusable after an io_uring submission or completion failure");
        return reader->plan.failure;
    }
    reader->active_weights = dst_weights;
    reader->active_scales = dst_scales;
    uint64_t planning = now_ns();
    bool planned = ple_plan(&reader->plan, ids, count, dst_weights, dst_scales, dst_is_cached);
    uint64_t plan_elapsed = now_ns() - planning;
    reader->step_plan_ns += plan_elapsed ? plan_elapsed : 1;
    if (planned && reader->plan.job_count) ple_uring_run(reader);
    pthread_mutex_lock(&reader->mutex);
    reader->total_read_calls += reader->plan.read_calls;
    reader->total_unique_blocks += reader->plan.unique_blocks;
    reader->total_read_bytes += reader->plan.read_bytes;
    pthread_mutex_unlock(&reader->mutex);
    return reader->plan.failure;
}

static failure_t read_rows(reader_t *reader, const char *ids, uint32_t count) {
    return read_rows_into(reader, ids, count, reader->weights, reader->scales, false);
}

/* --- worker ------------------------------------------------------------------ */

/* Wait until the GPU has written ``target`` (or later) into control word ``word``: spin
 * for 200 us (a step the GPU is already on answers within it), then poll every 10 us. */
static bool wait_for_gpu(reader_t *reader, int word, uint32_t target, failure_t *failure,
                         const char *what, const char *hint) {
    uint64_t start = now_ns();
    for (;;) {
        if (reached(__atomic_load_n(&reader->control[word], __ATOMIC_ACQUIRE), target))
            return true;
        if (__atomic_load_n(&reader->closing, __ATOMIC_RELAXED)) {
            snprintf(failure->message, sizeof(failure->message), "Engram reader closed before %s %u", what,
                     target);
            return false;
        }
        if (now_ns() - start > reader->ids_timeout_ns) {
            snprintf(failure->message, sizeof(failure->message), "%s %u did not come within %.1f s: %s", what,
                     target, (double)reader->ids_timeout_ns * 1e-9, hint);
            return false;
        }
        if (now_ns() - start > 200000) {
            struct timespec pause = {0, 10000};
            nanosleep(&pause, NULL);
        } else {
#if defined(__aarch64__)
            __asm__ volatile("yield");
#endif
        }
    }
}

static void pause_for_delay(reader_t *reader) {
    if (reader->delay_ns) {
        struct timespec pause = {(time_t)(reader->delay_ns / 1000000000ull),
                                 (long)(reader->delay_ns % 1000000000ull)};
        nanosleep(&pause, NULL);
    }
}

/* The step's early part: when the GPU has already gathered the previous step (>99% of
 * steps), read directly into mapped weights/scales with zero bounce buffer or extra copy.
 * Only when the previous gather is still in flight, stage into private early buffers
 * immediately (overlapping NVMe I/O with GPU execution) and copy once consumed. */
static bool read_early(reader_t *reader, step_t step, failure_t *failure, double *consumed_wait,
                       double *early_read, uint64_t *early_count) {
    /* Sequences run 1, 2, ... 0xFFFFFFFF, 1: the step before 1 is 0xFFFFFFFF (or none,
     * when the control block's 0 has reached it already). */
    uint32_t previous = step.sequence == 1 ? 0xFFFFFFFFu : step.sequence - 1;
    bool early_match = false;
    if (reader->prefetched_count > 0 && reader->prefetched_count <= step.count &&
        reader->prefetched_early &&
        memcmp(reader->prefetched_early, step.early,
               (size_t)reader->prefetched_count * sizeof(*step.early)) == 0) {
        early_match = true;
        for (uint32_t i = reader->prefetched_count; i < step.count; i++) {
            if (step.early[i] != -1) {
                early_match = false;
                break;
            }
        }
    }
    if (early_match) {
        uint32_t copy_count = reader->prefetched_count;
        reader->prefetched_count = 0;
        pthread_mutex_lock(&reader->mutex);
        reader->early_prefetched_steps++;
        pthread_mutex_unlock(&reader->mutex);
        if (reached(__atomic_load_n(&reader->control[CTRL_CONSUMED_SEQ], __ATOMIC_ACQUIRE), previous)) {
            *consumed_wait = 0.0;
        } else {
            control_store(reader, CTRL_PHASE, PHASE_WAITING_CONSUMED);
            uint64_t wait_start = now_ns();
            if (!wait_for_gpu(reader, CTRL_CONSUMED_SEQ, previous, failure,
                              "the GPU's gather of the rows of Engram step",
                              "the GPU never gathered the previous step's rows"))
                return false;
            *consumed_wait = (double)(now_ns() - wait_start) * 1e-9;
        }
        uint64_t copy_start = now_ns();
        for (uint32_t i = 0; i < copy_count; i++) {
            int64_t id = step.early[i];
            if (id == UNKNOWN_ID) continue;
            (*early_count)++;
            if (id < reader->plan.tp_start || id >= reader->plan.tp_end ||
                !((reader->plan.columns >> (i % PLE_COLUMNS)) & 1u))
                continue;
            if (reader->plan.cache_sets) reader->plan.cache_hits++;
            memcpy(reader->weights + (size_t)i * 256u,
                   reader->early_weights + (size_t)i * 256u, 256u);
            memcpy(reader->scales + (size_t)i * 8u,
                   reader->early_scales + (size_t)i * 8u, 8u);
        }
        *early_read = (double)(now_ns() - copy_start) * 1e-9;
        return true;
    }
    reader->prefetched_count = 0;
    if (reached(__atomic_load_n(&reader->control[CTRL_CONSUMED_SEQ], __ATOMIC_ACQUIRE), previous)) {
        *consumed_wait = 0.0;
        uint64_t read_start = now_ns();
        pause_for_delay(reader);
        control_store(reader, CTRL_PHASE, PHASE_READING);
        *failure = read_rows(reader, (const char *)step.early, step.count);
        *early_read = (double)(now_ns() - read_start) * 1e-9;
        for (uint32_t i = 0; i < step.count; i++) *early_count += step.early[i] != UNKNOWN_ID;
        return !failure->message[0];
    }

    uint64_t read_start = now_ns();
    pause_for_delay(reader);
    control_store(reader, CTRL_PHASE, PHASE_READING);
    *failure = read_rows_into(reader, (const char *)step.early, step.count,
                              reader->early_weights, reader->early_scales, true);
    uint64_t read_done = now_ns();
    if (failure->message[0]) {
        *early_read = (double)(read_done - read_start) * 1e-9;
        return false;
    }
    pthread_mutex_lock(&reader->mutex);
    reader->early_prefetched_steps++;
    pthread_mutex_unlock(&reader->mutex);

    control_store(reader, CTRL_PHASE, PHASE_WAITING_CONSUMED);
    uint64_t wait_start = now_ns();
    if (!wait_for_gpu(reader, CTRL_CONSUMED_SEQ, previous, failure,
                      "the GPU's gather of the rows of Engram step",
                      "the GPU never gathered the previous step's rows"))
        return false;
    uint64_t consumed_at = now_ns();
    *consumed_wait = (double)(consumed_at - wait_start) * 1e-9;

    for (uint32_t i = 0; i < step.count; i++) {
        int64_t id = step.early[i];
        if (id == UNKNOWN_ID) continue;
        (*early_count)++;
        if (id < reader->plan.tp_start || id >= reader->plan.tp_end ||
            !((reader->plan.columns >> (i % PLE_COLUMNS)) & 1u))
            continue;
        memcpy(reader->weights + (size_t)i * 256u,
               reader->early_weights + (size_t)i * 256u, 256u);
        memcpy(reader->scales + (size_t)i * 8u,
               reader->early_scales + (size_t)i * 8u, 8u);
    }
    *early_read = (double)((read_done - read_start) + (now_ns() - consumed_at)) * 1e-9;
    return true;
}

/* The GPU's IDs: check the early ones (debug) and keep only those the host did not know. */
static bool late_ids(reader_t *reader, step_t step, failure_t *failure) {
    const int64_t *gpu = (const int64_t *)reader->ids;
    for (uint32_t i = 0; i < step.count; i++) {
        int64_t id;
        memcpy(&id, gpu + i, sizeof(id));
        if (step.early && step.early[i] != UNKNOWN_ID) {
            if (reader->check_early && step.early[i] != id) {
                snprintf(failure->message, sizeof(failure->message),
                         "Engram step %u lookup %u: the host hashed row %lld, the GPU row %lld", step.sequence,
                         i, (long long)step.early[i], (long long)id);
                return false;
            }
            reader->late[i] = -1;
        } else {
            reader->late[i] = id;
        }
    }
    return true;
}

static uint64_t drop_plan_page_cache(reader_t *reader) {
    if (!reader->drop_page_cache) {
        reader->plan.job_count = 0;
        return 0;
    }
    uint64_t bytes = 0;
    for (size_t i = 0; i < reader->plan.job_count; i++) {
        ple_job_t *job = &reader->plan.jobs[i];
        int fd = reader->plan.files[job->file].fd;
        if (posix_fadvise(fd, job->offset, job->length, POSIX_FADV_DONTNEED) == 0) {
            bytes += job->length;
        }
    }
    reader->plan.job_count = 0;
    return bytes;
}

static void serve(reader_t *reader, step_t step) {
    failure_t failure = {{0}};
    if (step.is_prefetch) {
        reader->prefetched_count = 0;
        if (step.early && reader->plan.cache_sets && reader->prefetched_early) {
            failure = read_rows_into(reader, (const char *)step.early, step.count,
                                     reader->early_weights, reader->early_scales, true);
            if (!failure.message[0] && !reader->poisoned) {
                ple_cache_commit(&reader->plan);
                memcpy(reader->prefetched_early, step.early,
                       (size_t)step.count * sizeof(*step.early));
                reader->prefetched_count = step.count;
            }
            (void)drop_plan_page_cache(reader);
        }
        pthread_mutex_lock(&reader->mutex);
        if (failure.message[0]) {
            if (!reader->failure.message[0]) reader->failure = failure;
        } else {
            reader->prefetch_steps++;
        }
        pthread_mutex_unlock(&reader->mutex);
        if (failure.message[0]) control_store(reader, CTRL_PHASE, PHASE_FAILED);
        return;
    }
    uint64_t started = now_ns();
    uint32_t gpu_sequence = 0, gpu_wait_ns = 0;
    reader->step_plan_ns = 0;
    reader->plan.job_count = 0;
    if (reader->timeline) {
        gpu_sequence = __atomic_load_n(&reader->control[CTRL_GPU_WAIT_SEQ], __ATOMIC_ACQUIRE);
        gpu_wait_ns = __atomic_load_n(&reader->control[CTRL_GPU_WAIT_NS], __ATOMIC_RELAXED);
    }
    control_store(reader, CTRL_PHASE_SEQ, step.sequence);
    uint64_t early_count = 0, fadvise_early = 0;
    double consumed_wait = 0.0, early_read = 0.0;
    bool ok = !step.early || read_early(reader, step, &failure, &consumed_wait, &early_read, &early_count);
    if (ok && step.early && step.has_unknown) {
        ple_cache_commit(&reader->plan);
        fadvise_early = drop_plan_page_cache(reader);
    }
    uint64_t start = now_ns(), ids_at = start, read_at = start;
    if (ok && (step.has_unknown || reader->check_early)) {
        control_store(reader, CTRL_PHASE, PHASE_WAITING_IDS);
        ok = wait_for_gpu(reader, CTRL_IDS_SEQ, step.sequence, &failure, "the row IDs of Engram step",
                          "the GPU is blocked before the step's Engram staging");
        ids_at = now_ns();
        if (ok && !step.early) pause_for_delay(reader);
        if (ok) ok = late_ids(reader, step, &failure);
        if (ok && step.has_unknown) {
            control_store(reader, CTRL_PHASE, PHASE_READING);
            failure = read_rows(reader, (const char *)reader->late, step.count);
        }
        read_at = now_ns();
    }
    if (failure.message[0]) {
        pthread_mutex_lock(&reader->mutex);
        if (!reader->failure.message[0]) reader->failure = failure;
        pthread_mutex_unlock(&reader->mutex);
        control_store(reader, CTRL_PHASE, PHASE_FAILED);
        return;
    }
    /* The rows are in place before the GPU can see the step. */
    __atomic_store_n(&reader->control[CTRL_ROWS_SEQ], step.sequence, __ATOMIC_RELEASE);
    control_store(reader, CTRL_PHASE, PHASE_IDLE);
    uint64_t published_at = now_ns();
    /* Off-critical-path cache population: the GPU has already been woken via CTRL_ROWS_SEQ. */
    ple_cache_commit(&reader->plan);
    uint64_t fadvise_late = drop_plan_page_cache(reader);
    pthread_mutex_lock(&reader->mutex);
    reader->steps_done++;
    reader->lookups_done += step.count;
    reader->early_lookups += early_count;
    reader->fadvise_bytes += fadvise_early + fadvise_late;
    reader->consumed_wait_seconds += consumed_wait;
    reader->early_read_seconds += early_read;
    reader->ids_wait_seconds += (double)(ids_at - start) * 1e-9;
    reader->read_seconds += (double)(read_at - ids_at) * 1e-9;
    if (reader->timeline) {
        record_t record = {step.sequence, step.count, (uint32_t)early_count, gpu_sequence, gpu_wait_ns,
                           step.submitted, started, started + (uint64_t)(consumed_wait * 1e9),
                           ids_at, read_at, published_at, reader->step_plan_ns};
        reader->timeline[reader->timeline_next++ % TIMELINE] = record;
    }
    pthread_mutex_unlock(&reader->mutex);
}

/* The process's one worker and its queue of (reader, step) jobs. */
typedef struct {
    reader_t *reader;
    step_t step;
} job_t;

static struct {
    pthread_mutex_t mutex;
    pthread_cond_t wake, space, idle;
    job_t jobs[QUEUE];
    uint64_t head, tail; /* tail - head jobs are queued */
    bool started;
} queue = {.mutex = PTHREAD_MUTEX_INITIALIZER};

static pthread_once_t queue_once = PTHREAD_ONCE_INIT;

static void queue_init(void) {
    pthread_condattr_t monotonic;
    pthread_condattr_init(&monotonic);
    pthread_condattr_setclock(&monotonic, CLOCK_MONOTONIC);
    pthread_cond_init(&queue.wake, &monotonic);
    pthread_cond_init(&queue.space, &monotonic);
    pthread_cond_init(&queue.idle, &monotonic);
    pthread_condattr_destroy(&monotonic);
}

static unsigned worker_l2_kib;

static unsigned read_cpu_l2_kib(int cpu) {
    char path[128], buf[32];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index2/size", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return 0;
    buf[n] = '\0';
    char *end = NULL;
    unsigned long val = strtoul(buf, &end, 10);
    if (end == buf || val == 0) return 0;
    while (*end == ' ' || *end == '\t') end++;
    if (*end == 'M' || *end == 'm') return (unsigned)(val * 1024ul);
    if (*end == 'G' || *end == 'g') return (unsigned)(val * 1024ul * 1024ul);
    return (unsigned)val;
}

static int parse_cpu_list_mask(const char *spec, const cpu_set_t *allowed, cpu_set_t *out) {
    CPU_ZERO(out);
    if (!spec || !*spec) return 0;
    const char *p = spec;
    int count = 0;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t') p++;
        char *end = NULL;
        long start = strtol(p, &end, 10);
        if (end == p || start < 0 || start >= CPU_SETSIZE) return 0;
        long stop = start;
        p = end;
        if (*p == '-') {
            p++;
            stop = strtol(p, &end, 10);
            if (end == p || stop < start || stop >= CPU_SETSIZE) return 0;
            p = end;
        }
        for (long c = start; c <= stop; c++) {
            if (!allowed || CPU_ISSET((int)c, allowed)) {
                if (!CPU_ISSET((int)c, out)) {
                    CPU_SET((int)c, out);
                    count++;
                }
            }
        }
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') p++;
        else if (*p != '\0') return 0;
    }
    return count;
}

static void pin_worker_to_max_l2_cpus(void) {
    pthread_setname_np(pthread_self(), "engram-reader");
    const char *env = getenv("SPARK_DS41F_ENGRAM_CPU_AFFINITY");
    if (env && strcmp(env, "0") == 0) return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    if (env && *env && strcmp(env, "1") != 0 && strcmp(env, "auto") != 0) {
        cpu_set_t explicit_set;
        if (parse_cpu_list_mask(env, &allowed, &explicit_set) > 0 ||
            parse_cpu_list_mask(env, NULL, &explicit_set) > 0) {
            if (sched_setaffinity(0, sizeof(explicit_set), &explicit_set) == 0) {
                unsigned max_kib = 0;
                for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
                    if (!CPU_ISSET(cpu, &explicit_set)) continue;
                    unsigned kib = read_cpu_l2_kib(cpu);
                    if (kib > max_kib) max_kib = kib;
                }
                if (max_kib > 0) {
                    __atomic_store_n(&worker_l2_kib, max_kib, __ATOMIC_RELAXED);
                }
                return;
            }
        }
    }
    unsigned max_kib = 0;
    cpu_set_t best;
    CPU_ZERO(&best);
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        unsigned kib = read_cpu_l2_kib(cpu);
        if (kib > max_kib) {
            max_kib = kib;
            CPU_ZERO(&best);
            CPU_SET(cpu, &best);
        } else if (kib > 0 && kib == max_kib) {
            CPU_SET(cpu, &best);
        }
    }
    if (max_kib > 0 && CPU_COUNT(&best) > 0) {
        if (sched_setaffinity(0, sizeof(best), &best) == 0) {
            __atomic_store_n(&worker_l2_kib, max_kib, __ATOMIC_RELAXED);
        }
    }
}

static void *worker_main(void *arg) {
    (void)arg;
    /* Sleep as asked: the default 50 us timer slack would stretch every 10 us poll. */
    prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
    pin_worker_to_max_l2_cpus();
    pthread_mutex_lock(&queue.mutex);
    for (;;) {
        while (queue.head == queue.tail) pthread_cond_wait(&queue.wake, &queue.mutex);
        job_t job = queue.jobs[queue.head % QUEUE];
        queue.head++;
        pthread_cond_broadcast(&queue.space);
        pthread_mutex_unlock(&queue.mutex);
        pthread_mutex_lock(&job.reader->mutex);
        bool failed = job.reader->failure.message[0] != 0;
        pthread_mutex_unlock(&job.reader->mutex);
        if (!failed) serve(job.reader, job.step);
        free(job.step.early);
        pthread_mutex_lock(&queue.mutex);
        job.reader->pending--;
        pthread_cond_broadcast(&queue.idle);
    }
    return NULL;
}

/* Close the reader: its queued steps fail at once, then it is freed. */
static void release(reader_t *reader) {
    if (!reader) return;
    pthread_mutex_lock(&queue.mutex);
    __atomic_store_n(&reader->closing, true, __ATOMIC_RELAXED);
    while (reader->pending) pthread_cond_wait(&queue.idle, &queue.mutex);
    pthread_mutex_unlock(&queue.mutex);
    if (reader->ring_ready) io_uring_queue_exit(&reader->ring);
    free(reader->iovecs);
    free(reader->late);
    free(reader->early_weights);
    free(reader->early_scales);
    free(reader->prefetched_early);
    free(reader->timeline);
    free(reader->free_slots);
    free(reader->registered_fds);
    ple_plan_release(&reader->plan);
    free(reader->buffers);
    free(reader->slots);
    pthread_mutex_destroy(&reader->mutex);
    free(reader);
}

static void delete_capsule(PyObject *capsule) {
    reader_t *reader = PyCapsule_GetPointer(capsule, READER_CAPSULE);
    if (reader) {
        Py_BEGIN_ALLOW_THREADS
        release(reader);
        Py_END_ALLOW_THREADS
    }
}

/* --- Python surface ---------------------------------------------------------- */

static reader_t *get_reader(PyObject *capsule) {
    return PyCapsule_GetPointer(capsule, READER_CAPSULE);
}

/* reader(table_rows, shard_start, shard_end, columns, max_lookups, queue_depth, ids, weights,
 *        scales, control, ids_timeout_s, delay_s, check_early, timeline[, cache_rows]): the rank
 * reads the rows in [shard_start, shard_end) of the hash columns in the 24-bit mask ``columns``
 * (lookup i is column i % 24); the four buffers are addresses of host memory that outlives the
 * reader; check_early (debug) fails a step whose early ID differs from the GPU's; timeline
 * (debug) records each served step. */
static PyObject *py_reader(PyObject *self, PyObject *args) {
    (void)self;
    long long table_rows, shard_start, shard_end, max_lookups;
    unsigned int columns;
    int depth;
    unsigned long long ids, weights, scales, control;
    double ids_timeout, delay;
    int check_early, timeline;
    long long cache_rows = PLE_DEFAULT_CACHE_ROWS;
    if (!PyArg_ParseTuple(args, "LLLILiKKKKddpp|L", &table_rows, &shard_start, &shard_end, &columns, &max_lookups,
                          &depth, &ids, &weights, &scales, &control, &ids_timeout, &delay, &check_early,
                          &timeline, &cache_rows))
        return NULL;
    if (table_rows <= 0 || shard_start < 0 || shard_end < shard_start || shard_end > table_rows ||
        !columns || (columns & ~PLE_ALL_COLUMNS) || cache_rows < 0 ||
        max_lookups <= 0 || max_lookups > UINT32_MAX || depth <= 0 || depth > 32768 || !ids ||
        !weights || !scales || !control || !(ids_timeout > 0) || delay < 0)
        return PyErr_Format(PyExc_ValueError, "invalid Engram reader geometry, buffers or times");
    reader_t *reader = calloc(1, sizeof(*reader));
    if (!reader) return PyErr_NoMemory();
    pthread_mutex_init(&reader->mutex, NULL);
    /* One whole-table source per plane: shard 0 spans every row. */
    int error = ple_plan_init(&reader->plan, table_rows, table_rows, shard_start, shard_end, columns, 256, 8,
                              max_lookups, cache_rows);
    if (error) {
        release(reader);
        return error == ENOMEM ? PyErr_NoMemory()
                               : PyErr_Format(PyExc_ValueError, "Engram row plan: %s", strerror(error));
    }
    reader->ids = (const char *)(uintptr_t)ids;
    reader->weights = (char *)(uintptr_t)weights;
    reader->scales = (char *)(uintptr_t)scales;
    reader->control = (uint32_t *)(uintptr_t)control;
    reader->ids_timeout_ns = (uint64_t)(ids_timeout * 1e9);
    reader->delay_ns = (uint64_t)(delay * 1e9);
    reader->check_early = check_early != 0;
    const char *drop_env = getenv("SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE");
    reader->drop_page_cache = !drop_env || strcmp(drop_env, "0") != 0;
    if (timeline && !(reader->timeline = calloc(TIMELINE, sizeof(*reader->timeline)))) {
        release(reader);
        return PyErr_NoMemory();
    }
    reader->late = malloc((size_t)max_lookups * sizeof(*reader->late));
    reader->early_weights = malloc((size_t)max_lookups * 256);
    reader->early_scales = malloc((size_t)max_lookups * 8);
    reader->prefetched_early = malloc((size_t)max_lookups * sizeof(*reader->prefetched_early));
    if (!reader->late || !reader->early_weights || !reader->early_scales || !reader->prefetched_early) {
        release(reader);
        return PyErr_NoMemory();
    }
    reader->slots_count = (unsigned)depth;
    reader->slots = calloc(reader->slots_count, sizeof(*reader->slots));
    error = posix_memalign((void **)&reader->buffers, PLE_BLOCK, (size_t)reader->slots_count * PLE_READ_MAX);
    if (!reader->slots || error) {
        release(reader);
        return PyErr_NoMemory();
    }
    for (unsigned i = 0; i < reader->slots_count; i++)
        reader->slots[i].buffer = reader->buffers + (size_t)i * PLE_READ_MAX;
    error = io_uring_queue_init(reader->slots_count, &reader->ring, 0);
    if (error < 0) {
        release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring initialization failed: %s; enable io_uring in the kernel and "
                            "container policy",
                            strerror(-error));
    }
    reader->ring_ready = true;
    reader->iovecs = calloc(reader->slots_count, sizeof(*reader->iovecs));
    reader->free_slots = calloc(reader->slots_count, sizeof(*reader->free_slots));
    if (!reader->iovecs || !reader->free_slots) {
        release(reader);
        return PyErr_NoMemory();
    }
    for (unsigned i = 0; i < reader->slots_count; i++) {
        reader->iovecs[i].iov_base = reader->slots[i].buffer;
        reader->iovecs[i].iov_len = PLE_READ_MAX;
    }
    error = io_uring_register_buffers(&reader->ring, reader->iovecs, reader->slots_count);
    if (error < 0) {
        release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring buffer registration failed: %s; raise the locked-memory limit "
                            "or reduce queue_depth",
                            strerror(-error));
    }
    PyObject *capsule = PyCapsule_New(reader, READER_CAPSULE, delete_capsule);
    if (!capsule) release(reader);
    return capsule;
}

/* add(reader, path, offset, scale): register the table's weight or scale plane. */
static PyObject *py_add(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    const char *path;
    long long offset;
    int scale;
    if (!PyArg_ParseTuple(args, "OsLp", &capsule, &path, &offset, &scale)) return NULL;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    int error;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->mutex);
    if (__atomic_load_n(&reader->started, __ATOMIC_RELAXED)) {
        error = EBUSY;
    } else {
        reader->plan.failure.message[0] = 0;
        error = ple_add_source(&reader->plan, 0, path, offset, scale != 0, ple_register_files, reader);
        failure = reader->plan.failure;
    }
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    if (error == EBUSY) return PyErr_Format(PyExc_RuntimeError, "Engram sources are fixed once steps run");
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    if (error) return PyErr_Format(PyExc_ValueError, "Engram source: %s", strerror(error));
    Py_RETURN_NONE;
}

static bool extract_col_array(PyObject *obj, int64_t out[PLE_COLUMNS], bool *present) {
    if (obj == Py_None) {
        *present = false;
        return true;
    }
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_C_CONTIGUOUS) < 0) return false;
    if (view.len != (Py_ssize_t)(PLE_COLUMNS * sizeof(int64_t))) {
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_ValueError, "Engram column map array must be %u int64 values", PLE_COLUMNS);
        return false;
    }
    memcpy(out, view.buf, PLE_COLUMNS * sizeof(int64_t));
    PyBuffer_Release(&view);
    *present = true;
    return true;
}

/* add_packed(reader, path, offset[, col_start, col_end, col_delta]): register a single-plane
 * 4 KiB-page-aligned (15 x 264B rows per 4096B page) row-interleaved source. */
static PyObject *py_add_packed(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule, *start_obj = Py_None, *end_obj = Py_None, *delta_obj = Py_None;
    const char *path;
    long long offset;
    if (!PyArg_ParseTuple(args, "OsL|OOO", &capsule, &path, &offset, &start_obj, &end_obj, &delta_obj))
        return NULL;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    int64_t col_start[PLE_COLUMNS] = {0}, col_end[PLE_COLUMNS] = {0}, col_delta[PLE_COLUMNS] = {0};
    bool has_start = false, has_end = false, has_delta = false;
    if (!extract_col_array(start_obj, col_start, &has_start) ||
        !extract_col_array(end_obj, col_end, &has_end) ||
        !extract_col_array(delta_obj, col_delta, &has_delta))
        return NULL;
    if (has_start != has_end || has_start != has_delta)
        return PyErr_Format(PyExc_ValueError, "Engram column map requires col_start, col_end and col_delta together");
    int error;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->mutex);
    if (__atomic_load_n(&reader->started, __ATOMIC_RELAXED)) {
        error = EBUSY;
    } else {
        reader->plan.failure.message[0] = 0;
        error = ple_add_packed_source(&reader->plan, 0, path, offset,
                                      has_start ? col_start : NULL,
                                      has_end ? col_end : NULL,
                                      has_delta ? col_delta : NULL,
                                      ple_register_files, reader);
        failure = reader->plan.failure;
    }
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    if (error == EBUSY) return PyErr_Format(PyExc_RuntimeError, "Engram sources are fixed once steps run");
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    if (error) return PyErr_Format(PyExc_ValueError, "Engram packed source: %s", strerror(error));
    Py_RETURN_NONE;
}

static bool pread_exact(int fd, char *buf, size_t len, off_t offset, failure_t *failure, const char *what) {
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, buf + done, len - done, offset + (off_t)done);
        if (n < 0) {
            if (errno == EINTR) continue;
            system_error(failure, what);
            return false;
        }
        if (n == 0) {
            snprintf(failure->message, sizeof(failure->message), "%s: unexpected end of file", what);
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

static bool pwrite_exact(int fd, const char *buf, size_t len, off_t offset, failure_t *failure, const char *what) {
    size_t done = 0;
    while (done < len) {
        ssize_t n = pwrite(fd, buf + done, len - done, offset + (off_t)done);
        if (n < 0) {
            if (errno == EINTR) continue;
            system_error(failure, what);
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

/* pack_columns(weight_path, weight_offset, scale_path, scale_offset, out_path, col_starts, col_rows):
 * stream the owned columns' 256-byte E4M3 weights and 8-byte E8M0 scales into a 4 KiB-aligned
 * 15-row-per-page (15 * 264 = 3,960 B + 136 B padding per 4096 B page) file at out_path,
 * dropping OS page-cache pages after each chunk so unified memory stays free. */
static PyObject *py_pack_columns(PyObject *self, PyObject *args) {
    (void)self;
    const char *w_path, *s_path, *out_path;
    long long w_offset, s_offset;
    PyObject *starts_obj, *rows_obj;
    if (!PyArg_ParseTuple(args, "sLsLsOO", &w_path, &w_offset, &s_path, &s_offset, &out_path,
                          &starts_obj, &rows_obj))
        return NULL;
    if (w_offset < 0 || s_offset < 0)
        return PyErr_Format(PyExc_ValueError, "Engram source offsets must be non-negative");
    Py_buffer starts_view, rows_view;
    if (PyObject_GetBuffer(starts_obj, &starts_view, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(rows_obj, &rows_view, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&starts_view);
        return NULL;
    }
    if (starts_view.len <= 0 || starts_view.len != rows_view.len ||
        starts_view.len % (Py_ssize_t)sizeof(int64_t) != 0) {
        PyBuffer_Release(&starts_view);
        PyBuffer_Release(&rows_view);
        return PyErr_Format(PyExc_ValueError, "col_starts and col_rows must be equal-length non-empty int64 arrays");
    }
    size_t num_cols = (size_t)starts_view.len / sizeof(int64_t);
    int64_t *col_starts = malloc(num_cols * sizeof(int64_t));
    int64_t *col_rows = malloc(num_cols * sizeof(int64_t));
    if (!col_starts || !col_rows) {
        free(col_starts);
        free(col_rows);
        PyBuffer_Release(&starts_view);
        PyBuffer_Release(&rows_view);
        return PyErr_NoMemory();
    }
    memcpy(col_starts, starts_view.buf, num_cols * sizeof(int64_t));
    memcpy(col_rows, rows_view.buf, num_cols * sizeof(int64_t));
    PyBuffer_Release(&starts_view);
    PyBuffer_Release(&rows_view);

    size_t chunk_pages = 4096u;
    size_t chunk_rows = chunk_pages * PLE_PACKED_ROWS_PER_PAGE;
    char *w_buf = malloc(chunk_rows * 256u);
    char *s_buf = malloc(chunk_rows * 8u);
    char *p_buf = calloc(chunk_pages, PLE_BLOCK);
    size_t tmp_len = strlen(out_path) + 32u;
    char *tmp_path = malloc(tmp_len);
    if (!w_buf || !s_buf || !p_buf || !tmp_path) {
        free(col_starts);
        free(col_rows);
        free(w_buf);
        free(s_buf);
        free(p_buf);
        free(tmp_path);
        return PyErr_NoMemory();
    }
    snprintf(tmp_path, tmp_len, "%s.tmp.%d", out_path, (int)getpid());
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    int w_fd = open(w_path, O_RDONLY | O_CLOEXEC);
    int s_fd = open(s_path, O_RDONLY | O_CLOEXEC);
    int out_fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (w_fd < 0) system_error(&failure, "open weight source");
    else if (s_fd < 0) system_error(&failure, "open scale source");
    else if (out_fd < 0) system_error(&failure, "open packed output");
    off_t out_off = 0;
    size_t buffered_rows = 0;
    for (size_t c = 0; c < num_cols && !failure.message[0]; c++) {
        int64_t start = col_starts[c], rows = col_rows[c];
        if (start < 0 || rows < 0) {
            snprintf(failure.message, sizeof(failure.message), "negative column start or row count");
            break;
        }
        int64_t done = 0;
        while (done < rows && !failure.message[0]) {
            size_t space = chunk_rows - buffered_rows;
            size_t rem = (size_t)(rows - done);
            size_t n = rem > space ? space : rem;
            off_t w_off = (off_t)w_offset + (off_t)(start + done) * 256;
            off_t s_off = (off_t)s_offset + (off_t)(start + done) * 8;
            if (!pread_exact(w_fd, w_buf, n * 256u, w_off, &failure, "read weight column") ||
                !pread_exact(s_fd, s_buf, n * 8u, s_off, &failure, "read scale column"))
                break;
            for (size_t r = 0; r < n; r++) {
                size_t phys = buffered_rows + r;
                char *dst = p_buf + (phys / PLE_PACKED_ROWS_PER_PAGE) * PLE_BLOCK +
                            (phys % PLE_PACKED_ROWS_PER_PAGE) * PLE_PACKED_ROW_BYTES;
                memcpy(dst, w_buf + r * 256u, 256u);
                memcpy(dst + 256u, s_buf + r * 8u, 8u);
            }
            buffered_rows += n;
            (void)posix_fadvise(w_fd, w_off, (off_t)(n * 256u), POSIX_FADV_DONTNEED);
            (void)posix_fadvise(s_fd, s_off, (off_t)(n * 8u), POSIX_FADV_DONTNEED);
            if (buffered_rows == chunk_rows) {
                size_t write_len = chunk_pages * PLE_BLOCK;
                if (!pwrite_exact(out_fd, p_buf, write_len, out_off, &failure, "write packed column"))
                    break;
                (void)posix_fadvise(out_fd, out_off, (off_t)write_len, POSIX_FADV_DONTNEED);
                out_off += (off_t)write_len;
                buffered_rows = 0;
            }
            done += (int64_t)n;
        }
    }
    if (!failure.message[0] && buffered_rows > 0) {
        size_t tail_pages = (buffered_rows + PLE_PACKED_ROWS_PER_PAGE - 1u) / PLE_PACKED_ROWS_PER_PAGE;
        size_t last_page_rows = buffered_rows % PLE_PACKED_ROWS_PER_PAGE;
        if (last_page_rows > 0) {
            char *last_page = p_buf + (tail_pages - 1u) * PLE_BLOCK;
            memset(last_page + last_page_rows * PLE_PACKED_ROW_BYTES, 0,
                   PLE_BLOCK - last_page_rows * PLE_PACKED_ROW_BYTES);
        }
        size_t write_len = tail_pages * PLE_BLOCK;
        if (pwrite_exact(out_fd, p_buf, write_len, out_off, &failure, "write packed column tail")) {
            (void)posix_fadvise(out_fd, out_off, (off_t)write_len, POSIX_FADV_DONTNEED);
            out_off += (off_t)write_len;
        }
    }
    if (!failure.message[0]) {
        if (fsync(out_fd) != 0) system_error(&failure, "fsync packed output");
        else (void)posix_fadvise(out_fd, 0, 0, POSIX_FADV_DONTNEED);
    }
    if (w_fd >= 0) close(w_fd);
    if (s_fd >= 0) close(s_fd);
    if (out_fd >= 0) close(out_fd);
    if (!failure.message[0]) {
        if (rename(tmp_path, out_path) != 0) system_error(&failure, "rename packed output");
    }
    if (failure.message[0]) unlink(tmp_path);
    Py_END_ALLOW_THREADS
    free(col_starts);
    free(col_rows);
    free(w_buf);
    free(s_buf);
    free(p_buf);
    free(tmp_path);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}

/* has_unknown(early): whether any int64 element of ``early`` equals UNKNOWN_ID. */
static PyObject *py_has_unknown(PyObject *self, PyObject *early_object) {
    (void)self;
    Py_buffer view;
    if (PyObject_GetBuffer(early_object, &view, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (view.len < 0 || view.len % (Py_ssize_t)sizeof(int64_t) != 0) {
        PyBuffer_Release(&view);
        return PyErr_Format(PyExc_ValueError, "Engram early IDs must be int64 values");
    }
    const int64_t *early = (const int64_t *)view.buf;
    size_t count = (size_t)view.len / sizeof(int64_t);
    bool found = false;
    for (size_t i = 0; i < count; i++) {
        if (early[i] == UNKNOWN_ID) {
            found = true;
            break;
        }
    }
    PyBuffer_Release(&view);
    return PyBool_FromLong(found);
}

/* submit(reader, sequence, count, wait_s, early, has_unknown): queue one step behind every
 * step already queued. ``early`` is None or ``count`` int64 IDs the host hashed
 * (UNKNOWN_ID where only the GPU knows the ID), copied here. Returns at once unless
 * the queue is full; then it waits for space, without the GIL. */
static PyObject *py_submit(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule, *early_object = Py_None;
    unsigned int sequence, count;
    double wait;
    int has_unknown_arg = -1;
    uint64_t submitted = now_ns();
    if (!PyArg_ParseTuple(args, "OIId|Oi", &capsule, &sequence, &count, &wait, &early_object,
                          &has_unknown_arg))
        return NULL;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    if (count > reader->plan.max_lookups)
        return PyErr_Format(PyExc_ValueError, "Engram step of %u lookups exceeds the reader's capacity", count);
    int64_t *early = NULL;
    bool has_unknown = true;
    if (early_object != Py_None) {
        Py_buffer view;
        if (PyObject_GetBuffer(early_object, &view, PyBUF_C_CONTIGUOUS) < 0) return NULL;
        if (view.len != (Py_ssize_t)count * (Py_ssize_t)sizeof(int64_t)) {
            PyBuffer_Release(&view);
            return PyErr_Format(PyExc_ValueError, "Engram early IDs must be %u int64 values", count);
        }
        early = malloc((size_t)count * sizeof(*early) + 1);
        if (!early) {
            PyBuffer_Release(&view);
            return PyErr_NoMemory();
        }
        memcpy(early, view.buf, (size_t)count * sizeof(*early));
        PyBuffer_Release(&view);
        if (has_unknown_arg >= 0) {
            has_unknown = has_unknown_arg != 0;
        } else {
            has_unknown = false;
            for (uint32_t i = 0; i < count && !has_unknown; i++) has_unknown = early[i] == UNKNOWN_ID;
        }
    }
    const char *problem = NULL;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += (time_t)wait;
    deadline.tv_nsec += (long)((wait - (double)(time_t)wait) * 1e9);
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&queue.mutex);
    for (;;) {
        pthread_mutex_lock(&reader->mutex);
        failure = reader->failure;
        pthread_mutex_unlock(&reader->mutex);
        if (failure.message[0] || queue.tail - queue.head < QUEUE) break;
        if (pthread_cond_timedwait(&queue.space, &queue.mutex, &deadline) == ETIMEDOUT) break;
    }
    if (!failure.message[0]) {
        if (queue.tail - queue.head >= QUEUE) {
            problem = "Engram readers fell more than 64 steps behind";
        } else if (!queue.started) {
            pthread_t thread;
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
            if (pthread_create(&thread, &attr, worker_main, NULL)) problem = "Engram reader thread did not start";
            else queue.started = true;
            pthread_attr_destroy(&attr);
        }
        if (!problem) {
            __atomic_store_n(&reader->started, true, __ATOMIC_RELAXED);
            queue.jobs[queue.tail % QUEUE] = (job_t){reader, {sequence, count, early, has_unknown, false, submitted}};
            early = NULL;
            queue.tail++;
            reader->pending++;
            pthread_cond_signal(&queue.wake);
        }
    }
    pthread_mutex_unlock(&queue.mutex);
    Py_END_ALLOW_THREADS
    free(early); /* not queued */
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    if (problem) return PyErr_Format(PyExc_RuntimeError, "%s", problem);
    Py_RETURN_NONE;
}

/* prefetch(reader, ids, wait_s): queue a background cache warm-up for ``ids`` (contiguous
 * int64 buffer) without advancing or touching the GPU step buffers or control block. */
static PyObject *py_prefetch(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule, *ids_object;
    double wait = 60.0;
    uint64_t submitted = now_ns();
    if (!PyArg_ParseTuple(args, "OO|d", &capsule, &ids_object, &wait)) return NULL;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(ids_object, &view, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (view.len < 0 || view.len % (Py_ssize_t)sizeof(int64_t) != 0) {
        PyBuffer_Release(&view);
        return PyErr_Format(PyExc_ValueError, "Engram prefetch IDs must be int64 values");
    }
    size_t total = (size_t)view.len / sizeof(int64_t);
    if (total == 0 || !reader->plan.cache_sets) {
        PyBuffer_Release(&view);
        Py_RETURN_NONE;
    }
    if (total > reader->plan.max_lookups) {
        PyBuffer_Release(&view);
        return PyErr_Format(PyExc_ValueError, "Engram prefetch of %zu lookups exceeds capacity", total);
    }
    int64_t *early = malloc(total * sizeof(*early));
    if (!early) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    memcpy(early, view.buf, total * sizeof(*early));
    PyBuffer_Release(&view);

    const char *problem = NULL;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += (time_t)wait;
    deadline.tv_nsec += (long)((wait - (double)(time_t)wait) * 1e9);
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&queue.mutex);
    for (;;) {
        pthread_mutex_lock(&reader->mutex);
        failure = reader->failure;
        pthread_mutex_unlock(&reader->mutex);
        if (failure.message[0] || queue.tail - queue.head < QUEUE) break;
        if (pthread_cond_timedwait(&queue.space, &queue.mutex, &deadline) == ETIMEDOUT) break;
    }
    if (!failure.message[0]) {
        if (queue.tail - queue.head >= QUEUE) {
            problem = "Engram readers fell more than 64 steps behind";
        } else if (!queue.started) {
            pthread_t thread;
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
            if (pthread_create(&thread, &attr, worker_main, NULL)) problem = "Engram reader thread did not start";
            else queue.started = true;
            pthread_attr_destroy(&attr);
        }
        if (!problem) {
            __atomic_store_n(&reader->started, true, __ATOMIC_RELAXED);
            queue.jobs[queue.tail % QUEUE] = (job_t){reader, {0, (uint32_t)total, early, false, true, submitted}};
            early = NULL;
            queue.tail++;
            reader->pending++;
            pthread_cond_signal(&queue.wake);
        }
    }
    pthread_mutex_unlock(&queue.mutex);
    Py_END_ALLOW_THREADS
    free(early);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    if (problem) return PyErr_Format(PyExc_RuntimeError, "%s", problem);
    Py_RETURN_NONE;
}

/* status(reader): the first failure (or None) and the worker's counters. */
static PyObject *py_status(PyObject *self, PyObject *capsule) {
    (void)self;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    failure_t failure;
    uint64_t steps, lookups, queued, early, fadvise_bytes, early_prefetched, prefetches;
    uint64_t cache_hits, cache_misses, read_calls, unique_blocks, read_bytes;
    double ids_wait, read, consumed_wait, early_read;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&queue.mutex);
    queued = reader->pending;
    pthread_mutex_unlock(&queue.mutex);
    pthread_mutex_lock(&reader->mutex);
    failure = reader->failure;
    steps = reader->steps_done;
    lookups = reader->lookups_done;
    ids_wait = reader->ids_wait_seconds;
    read = reader->read_seconds;
    early = reader->early_lookups;
    fadvise_bytes = reader->fadvise_bytes;
    early_prefetched = reader->early_prefetched_steps;
    prefetches = reader->prefetch_steps;
    cache_hits = reader->plan.cache_hits;
    cache_misses = reader->plan.cache_misses;
    read_calls = reader->total_read_calls;
    unique_blocks = reader->total_unique_blocks;
    read_bytes = reader->total_read_bytes;
    consumed_wait = reader->consumed_wait_seconds;
    early_read = reader->early_read_seconds;
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    PyObject *message = failure.message[0] ? PyUnicode_FromString(failure.message) : Py_NewRef(Py_None);
    if (!message) return NULL;
    unsigned l2_kib = __atomic_load_n(&worker_l2_kib, __ATOMIC_RELAXED);
    PyObject *result = Py_BuildValue(
        "{s:N,s:K,s:K,s:K,s:K,s:K,s:d,s:d,s:d,s:d,s:I,s:K,s:K,s:K,s:K,s:K,s:K,s:K}",
        "failure", message, "steps", (unsigned long long)steps, "lookups", (unsigned long long)lookups,
        "early_lookups", (unsigned long long)early, "queued", (unsigned long long)queued,
        "fadvise_bytes", (unsigned long long)fadvise_bytes, "ids_wait_seconds", ids_wait,
        "read_seconds", read, "consumed_wait_seconds", consumed_wait,
        "early_read_seconds", early_read, "worker_l2_kib", l2_kib,
        "early_prefetched_steps", (unsigned long long)early_prefetched,
        "prefetch_steps", (unsigned long long)prefetches,
        "cache_hits", (unsigned long long)cache_hits,
        "cache_misses", (unsigned long long)cache_misses,
        "read_calls", (unsigned long long)read_calls,
        "unique_blocks", (unsigned long long)unique_blocks,
        "read_bytes", (unsigned long long)read_bytes);
    return result;
}

/* timeline(reader) (debug): the records of the steps served since the last call (at most
 * the last TIMELINE), oldest first, as tuples in record_t's field order; [] when off. */
static PyObject *py_timeline(PyObject *self, PyObject *capsule) {
    (void)self;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    PyObject *result = PyList_New(0);
    if (!result || !reader->timeline) return result;
    record_t *copy = malloc(TIMELINE * sizeof(*copy));
    if (!copy) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }
    uint64_t first, last;
    pthread_mutex_lock(&reader->mutex);
    last = reader->timeline_next;
    first = reader->timeline_read + TIMELINE < last ? last - TIMELINE : reader->timeline_read;
    for (uint64_t i = first; i < last; i++) copy[i - first] = reader->timeline[i % TIMELINE];
    reader->timeline_read = last;
    pthread_mutex_unlock(&reader->mutex);
    for (uint64_t i = 0; i < last - first; i++) {
        record_t *r = &copy[i];
        PyObject *row = Py_BuildValue("(IIIIIKKKKKKK)", r->sequence, r->count, r->early, r->gpu_sequence,
                                      r->gpu_wait_ns, (unsigned long long)r->submitted,
                                      (unsigned long long)r->started, (unsigned long long)r->consumed,
                                      (unsigned long long)r->ids, (unsigned long long)r->read,
                                      (unsigned long long)r->published, (unsigned long long)r->plan_ns);
        if (!row || PyList_Append(result, row) < 0) {
            Py_XDECREF(row);
            Py_DECREF(result);
            free(copy);
            return NULL;
        }
        Py_DECREF(row);
    }
    free(copy);
    return result;
}

static PyMethodDef methods[] = {
    {"reader", py_reader, METH_VARARGS, NULL},
    {"add", py_add, METH_VARARGS, NULL},
    {"add_packed", py_add_packed, METH_VARARGS, NULL},
    {"pack_columns", py_pack_columns, METH_VARARGS, NULL},
    {"has_unknown", py_has_unknown, METH_O, NULL},
    {"submit", py_submit, METH_VARARGS, NULL},
    {"prefetch", py_prefetch, METH_VARARGS, NULL},
    {"status", py_status, METH_O, NULL},
    {"timeline", py_timeline, METH_O, NULL},
    {NULL, NULL, 0, NULL},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, .m_name = "_engram_rows_reader", .m_size = -1, .m_methods = methods,
};

PyMODINIT_FUNC PyInit__engram_rows_reader(void) {
    pthread_once(&queue_once, queue_init);
    PyObject *result = PyModule_Create(&module);
    if (!result) return NULL;
    if (PyModule_AddIntConstant(result, "ABI_VERSION", ABI_VERSION) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_IDS_SEQ", CTRL_IDS_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_CONSUMED_SEQ", CTRL_CONSUMED_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_ROWS_SEQ", CTRL_ROWS_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_TIMEOUT_US", CTRL_TIMEOUT_US) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_WAITED_US", CTRL_WAITED_US) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_FAILED_SEQ", CTRL_FAILED_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_GPU_WAIT_SEQ", CTRL_GPU_WAIT_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_GPU_WAIT_NS", CTRL_GPU_WAIT_NS) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_PHASE", CTRL_PHASE) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_PHASE_SEQ", CTRL_PHASE_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_WORDS", CTRL_WORDS) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
