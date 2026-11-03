/* Host payload-agreement control for Sparknet 1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf.
 * Compile with -I<exact-source>/tests/oneshot_sim -pthread and sanitizers.
 * The existing simulator includes the unchanged production _roce_proxy.c.
 * Only its verbs endpoint and allocation geometry are simulated here.
 * This exercises real post_op/stripe/flag logic, not the GPU reduction or NIC.
 */
#define main sparknet_upstream_simulator_main
/* Renaming upstream's unused main removes C's implicit return-0 exception. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#include "simulate.c"
#pragma GCC diagnostic pop
#undef main

enum { ROWS = 17, WIDTH = 5120, SHORT_ROWS = 6 };

static void payload_setup(void) {
    world = 3; ring4 = 0; stripes = 2; rotate = 0; exchange_mode = 0;
    setup(0);
    const size_t capacity = ((ROWS * WIDTH * 2 + 4095) / 4096) * 4096;
    uint64_t layout[7];
    assert(!roce_layout(world, capacity, layout));
    /* The original simulator uses 4 KiB slots. Retain its fake QPs/MR keys,
     * but resize the initialized regions to the actual 17x5120 BF16 payload. */
    for (int r = 0; r < world; r++) {
        roce_ctx_t *c = ranks[r];
        free(c->region);
        c->region = calloc(1, layout[4]);
        assert(c->region);
        c->slot_bytes = capacity; c->region_bytes = layout[4];
        c->recv_off = layout[0]; c->flag_off = layout[1];
        c->send_off = layout[2]; c->ctrl_off = layout[3];
    }
    roce_blob_t blobs[4];
    for (int r = 0; r < world; r++)
        assert(!roce_local_blob(ranks[r], &blobs[r], sizeof(blobs[r])));
    for (int r = 0; r < world; r++)
        assert(!roce_connect(ranks[r], blobs, sizeof(blobs)));
}

static void deliver_all(void) {
    /* No proxy threads in this narrow control. Preserve each fake reliable
     * QP's FIFO order using the upstream simulator's actual delivery helper. */
    for (int r = 0; r < world; r++) {
        roce_ctx_t *c = ranks[r];
        for (int h = 0; h < c->n_hca; h++) {
            for (int p = 0; p < world; p++) {
                struct ibv_qp *q = c->hca[h].qp[p];
                if (q) while (q->head) deliver(q);
            }
            assert(!drain_cq(c, h));
        }
    }
}

static void payload_case(int short_sender, int old_integer) {
    payload_setup();
    const uint16_t old_bits = old_integer ? 0xffffu : 0x40a0u; /* -1 integers / BF16 5 */
    for (uint32_t seq = 1; seq <= 3; seq++) {
        for (int r = 0; r < world; r++) {
            roce_ctx_t *c = ranks[r];
            unsigned rows = seq == 3 && r == 1 && short_sender ? SHORT_ROWS : ROWS;
            const unsigned bytes = rows * WIDTH * 2;
            uint16_t bits = seq == 1 ? old_bits : 0x40a0u;
            if (seq == 3) bits = r == 0 ? 0x3f80u : r == 1 ? 0x4000u : 0x4040u;
            uint16_t *send = (uint16_t *)(c->region + c->send_off + (seq & 1u) * c->slot_bytes);
            for (unsigned i = 0; i < bytes / 2; i++) send[i] = bits;
            assert(!post_op(c, seq, bytes));
        }
        deliver_all();
        for (int r = 0; r < world; r++)
            for (int p = 0; p < world; p++) if (p != r)
                for (int lane = 0; lane < receive_lanes(r, p); lane++)
                    assert(*flag(ranks[r], p, seq, lane) == seq);
    }
    roce_ctx_t *receiver = ranks[0];
    const uint16_t *received = (uint16_t *)(receiver->region + receiver->recv_off +
                                         ((size_t)1 * ROCE_SLOTS + 1) * receiver->slot_bytes);
    unsigned stale = 0, nan_bits = 0;
    for (unsigned i = 0; i < ROWS * WIDTH; i++) {
        const int in_tail = short_sender && i >= SHORT_ROWS * WIDTH;
        assert(received[i] == (in_tail ? old_bits : 0x4000u));
        stale += in_tail;
        nan_bits += (received[i] & 0x7f80u) == 0x7f80u && (received[i] & 0x007fu) != 0;
    }
    for (int r = 0; r < world; r++) {
        assert(!roce_failed(ranks[r]));
        assert(ranks[r]->ops_posted == 3);
    }
    printf("HOST-PAYLOAD-RESULT {\"short_sender\":%s,\"old_integer\":%s,"
           "\"receiver_rows\":17,\"sender_rows\":%u,\"row_width\":5120,"
           "\"sequence_flags\":3,\"stale_bf16_elements\":%u,\"nan_bit_patterns\":%u}\n",
           short_sender ? "true" : "false", old_integer ? "true" : "false",
           short_sender ? SHORT_ROWS : ROWS, stale, nan_bits);
    cleanup();
}

int main(void) {
    payload_case(0, 0);
    payload_case(1, 0);
    payload_case(0, 1);
    payload_case(1, 1);
    return 0;
}
