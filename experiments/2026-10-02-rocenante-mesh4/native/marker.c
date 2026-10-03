/* SPDX-License-Identifier: Apache-2.0
 * Adapted from SparkRing; see NOTICE. Only Ethernet EtherType is changed.
 * The marker is scoped to one IPv4 endpoint pair and reserved UDP source.
 * This is a hardware mutation, not an inventory command.
 */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/mlx5_api.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stop;
static void stopped(int sig) { (void)sig; stop = 1; }
int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: spark3-roce-marker RDMA_DEVICE SOURCE_IP DEST_IP\n"); return 2; }
    struct in_addr src, dst;
    if (inet_pton(AF_INET, argv[2], &src) != 1 || inet_pton(AF_INET, argv[3], &dst) != 1) return 2;
    struct sigaction sa = {.sa_handler = stopped};
    sigemptyset(&sa.sa_mask);
    pid_t parent = getppid();
    if (sigaction(SIGTERM, &sa, NULL) || sigaction(SIGINT, &sa, NULL) ||
        prctl(PR_SET_PDEATHSIG, SIGTERM) || parent == 1 || getppid() != parent) return 1;
    int count = 0, result = 1;
    struct ibv_device **devices = ibv_get_device_list(&count);
    struct ibv_context *ctx = NULL;
    if (!devices) return 1;
    for (int i = 0; i < count; i++)
        if (!strcmp(ibv_get_device_name(devices[i]), argv[1])) { ctx = ibv_open_device(devices[i]); break; }
    ibv_free_device_list(devices);
    if (!ctx) { perror("open RDMA device"); return 1; }
    struct mlx5dv_flow_match_parameters *mask = calloc(1, sizeof(*mask) + 0x180);
    struct mlx5dv_flow_match_parameters *value = calloc(1, sizeof(*value) + 0x180);
    struct mlx5dv_flow_matcher *matcher = NULL;
    struct ibv_flow_action *modify = NULL;
    struct ibv_flow *flow = NULL;
    if (!mask || !value) goto cleanup;
    mask->match_sz = value->match_sz = 0x180;
    /* mlx5 outer-header UDP source port occupies bytes 28..29. */
    ((uint8_t *)mask->match_buf)[28] = ((uint8_t *)mask->match_buf)[29] = 0xff;
    const uint16_t port = htons(65535);
    memcpy((uint8_t *)value->match_buf + 28, &port, sizeof(port));
    /* fte_match_set_lyr_2_4_bits from mlx5_ifc.h: EtherType byte 6,
     * IP protocol byte 16, UDP destination byte 30, IPv4 source/dest 44/60.
     * Scope to this diagonal so an unrelated neighbour QP whose UDP hash
     * happens to be 65535 cannot be marked (including NCCL on these HCAs).
     */
    const uint16_t ethertype = htons(0x0800), dport = htons(4791);
    memset((uint8_t *)mask->match_buf + 6, 0xff, 2);
    memcpy((uint8_t *)value->match_buf + 6, &ethertype, 2);
    ((uint8_t *)mask->match_buf)[16] = 0xff;
    ((uint8_t *)value->match_buf)[16] = 17;
    memset((uint8_t *)mask->match_buf + 30, 0xff, 2);
    memcpy((uint8_t *)value->match_buf + 30, &dport, 2);
    memset((uint8_t *)mask->match_buf + 44, 0xff, 4);
    memcpy((uint8_t *)value->match_buf + 44, &src, 4);
    memset((uint8_t *)mask->match_buf + 60, 0xff, 4);
    memcpy((uint8_t *)value->match_buf + 60, &dst, 4);
    struct mlx5dv_flow_matcher_attr attr = {
        .type = IBV_FLOW_ATTR_NORMAL, .priority = 0, .match_criteria_enable = 1,
        .match_mask = mask, .comp_mask = MLX5DV_FLOW_MATCHER_MASK_FT_TYPE,
        .ft_type = MLX5DV_FLOW_TABLE_TYPE_RDMA_TX,
    };
    matcher = mlx5dv_create_flow_matcher(ctx, &attr);
    if (!matcher) goto cleanup;
    /* SET outer EtherType field 0x03, width 16. 64-bit alignment is required. */
    union { uint64_t aligned; uint32_t words[2]; } command;
    command.words[0] = htonl((1U << 28) | (0x03U << 16) | 16U);
    command.words[1] = htonl(0x88b5);
    modify = mlx5dv_create_flow_action_modify_header(ctx, sizeof(command), &command.aligned,
                                                   MLX5DV_FLOW_TABLE_TYPE_RDMA_TX);
    if (!modify) goto cleanup;
    struct mlx5dv_flow_action_attr action = {.type = MLX5DV_FLOW_ACTION_IBV_FLOW_ACTION, .action = modify};
    flow = mlx5dv_create_flow(matcher, value, 1, &action);
    if (!flow) goto cleanup;
    printf("READY %s\n", argv[1]); fflush(stdout);
    while (!stop) {
        const struct timespec interval = {.tv_sec = 0, .tv_nsec = 100000000};
        nanosleep(&interval, NULL);
    }
    result = 0;
cleanup:
    if (result) perror("RDMA-TX marker");
    if (flow) ibv_destroy_flow(flow);
    if (modify) ibv_destroy_flow_action(modify);
    if (matcher) mlx5dv_destroy_flow_matcher(matcher);
    free(value); free(mask); ibv_close_device(ctx);
    return result;
}
