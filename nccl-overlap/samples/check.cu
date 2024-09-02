#include <stdio.h>
#include <stdlib.h>

/* Compile with
 *      nvcc streamPriorityRange.cu -o run
 *
 *  And run with:
 *      ./run
 *
 *  0 corresponds to the lowest priority (and the priority of the default stream)
 *  Negative numbers correspond to increasingly higher priority streams
 */

int main() {
	int low = -1, high = -1;

	cudaError_t ret = cudaDeviceGetStreamPriorityRange(&low, &high);

	printf("priority ranges from %d to %d\n", low, high);

	return 0;
}