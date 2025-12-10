#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Shared memory per block: %zu bytes (%zu KB)\n", 
           prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024);
    printf("Shared memory per SM: %zu bytes (%zu KB)\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024);
    
    return 0;
}
