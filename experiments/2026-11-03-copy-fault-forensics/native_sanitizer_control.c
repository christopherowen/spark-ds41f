/* Positive/negative controls for the host sanitizer tools, not proxy tests. */
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile int counter;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static void *increment(void *argument) {
    const int protected = *(const int *)argument;
    for (int i = 0; i < 2048; i++) {
        if (protected && pthread_mutex_lock(&mutex)) abort();
        counter++;
        if (protected && pthread_mutex_unlock(&mutex)) abort();
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "clean") && strcmp(argv[1], "asan") &&
                      strcmp(argv[1], "ubsan") && strcmp(argv[1], "tsan"))) {
        fprintf(stderr, "usage: %s clean|asan|ubsan|tsan\n", argv[0]);
        return 2;
    }
    volatile int *allocation = malloc(sizeof(int));
    if (!allocation) return 2;
    allocation[0] = 7;
    const int index = !strcmp(argv[1], "asan") ? 1 : 0;
    /* The bad read is one int beyond one allocated, initialized int. */
    const int observed = allocation[index];
    printf("read=%d\n", observed);
    free((void *)allocation);

    volatile int left = !strcmp(argv[1], "ubsan") ? INT_MAX : 7;
    volatile int right = 1;
    const int sum = left + right; /* Intentional signed overflow in ubsan mode. */
    printf("sum=%d\n", sum);

    const int protected = strcmp(argv[1], "tsan") != 0;
    pthread_t first, second;
    if (pthread_create(&first, NULL, increment, (void *)&protected)) return 2;
    if (pthread_create(&second, NULL, increment, (void *)&protected)) abort();
    if (pthread_join(first, NULL) || pthread_join(second, NULL)) abort();
    printf("counter=%d protected=%d\n", counter, protected);
    return 0;
}
