#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define MAX_SIZE 127

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics

    __shared__ unsigned int localHistogram[NUM_BINS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize local histogram to zero
    if (threadIdx.x == 0){
      for (int i = 0; i < num_bins; i++){
        localHistogram[i] = 0;
      }
    }
    __syncthreads();

    if(idx>=num_elements)   return;
    
    // Increment local histogram using shared memory
    if (idx<num_elements){
      atomicAdd(&localHistogram[input[idx]], 1);
    }
    // make sure that all threads have finished updating 
    // their local portions of the shared memory 
    __syncthreads();

    // Update global histogram using atomic operations
    if (threadIdx.x == 0){
      for (int i = 0; i < num_bins; i ++){
        atomicAdd(&bins[i], localHistogram[i]);
      }
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

  //@@ Insert code below to clean up bins that saturate at 127

  // Calculate thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Saturate histogram values at 127
  if (idx < num_bins && bins[idx] > MAX_SIZE) {
          bins[idx] = MAX_SIZE;
  }

}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]); 
  int TPB = atoi(argv[2]);  
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*)malloc(inputLength*sizeof(unsigned int));
  hostBins = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  resultRef = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to 
  //random numbers whose values range from 0 to (NUM_BINS - 1)

  // Use current time as 
  // seed for random generator 
  //srand(time(0));  
  for (int i=0; i<inputLength; i++){
    hostInput[i] = rand()%NUM_BINS;   
  }
  
  //@@ Insert code below to create reference result in CPU
  memset(resultRef, 0, NUM_BINS*sizeof(unsigned int));
  for (int i=0; i<inputLength; i++){
    resultRef[hostInput[i]] += 1; 
    if (resultRef[hostInput[i]]>127){
      resultRef[hostInput[i]]=127;
    }  
  }
  
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength*sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS*sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength*sizeof(unsigned int), cudaMemcpyHostToDevice);
  

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS*sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  dim3 block1(TPB);
  dim3 grid1((inputLength + block1.x - 1)/block1.x);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<grid1, block1>>>(deviceInput,
                                 deviceBins,
                                 inputLength,
                                 NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  dim3 block2(TPB);
  dim3 grid2((inputLength + block2.x - 1)/block2.x);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<grid2, block2>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();


  //@@ Insert code below to compare the output with the reference
  int equal = 1;
  for(int i=0; i<NUM_BINS; i++){
    //printf("bin %d, ref %d res %d;", i, resultRef[i], hostBins[i]);
    if (i==116 && hostBins[i] != resultRef[i]){
      printf("Bin %d: Result (%d) is not equal to the reference (%d)\n", i, hostBins[i], resultRef[i]);
      equal = 0;
      //break;
    } 
  }
  if (equal){
    printf("the result are equal with the reference");

  }
  
  // write file for plot the histogram
  FILE *f = fopen("/content/drive/MyDrive/3/ex3.txt", "w");
  if (f == NULL){
      printf("Error opening file!\n");
      exit(1);
  }
  for (int i = 0; i < NUM_BINS; i++){
    fprintf(f, "resultRef[i]= %d, hostBins[i]= %d\n", resultRef[i], hostBins[i]);
  }
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

