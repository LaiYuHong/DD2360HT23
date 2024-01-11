#include <stdio.h>
#include <sys/time.h>

#define DataType double
//#define nStreams 8

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  
  // acquire index first
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // vector addition
  if (id < len){
  	out[id] = in1[id]+in2[id];
  }
  
}

//@@ Insert code to implement timer start
void timeStart(struct timeval *startt)
{
  gettimeofday(startt,NULL);
}
//@@ Insert code to implement timer stop
void timeStop(struct timeval *startt)
{
  struct timeval stopt;
  float timet;
  gettimeofday(&stopt,NULL);
  timet = (stopt.tv_usec-startt->tv_usec)*1.0e-6 + stopt.tv_sec - startt->tv_sec;
  printf("Time: %f (s)\n",timet);
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;


  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  int nStreams = atoi(argv[3]);
  const int streamSize = inputLength/nStreams;
  const int streamBytes = streamSize * sizeof(DataType);
  
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  //cudaHostAlloc(&hostInput1, inputLength*sizeof(DataType), cudaHostAllocDefault);
  //cudaHostAlloc(&hostInput2, inputLength*sizeof(DataType), cudaHostAllocDefault);
  hostInput1 = (DataType *)malloc(inputLength*sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength*sizeof(DataType));
  hostOutput = (DataType *) malloc(inputLength*sizeof(DataType));
  resultRef = (DataType *) malloc(inputLength*sizeof(DataType));
  cudaStream_t stream[nStreams];
  for (int i=0; i<nStreams; ++i){
  	cudaStreamCreate(&stream[i]);
  }

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers
  //and create reference result in CPU
  
  // Use current time as 
  // seed for random generator 
  srand(time(0));  
  for (int i=0; i<inputLength; i++){
  	hostInput1[i] = rand();
  	hostInput2[i] = rand();
  	resultRef[i] = hostInput1[i]+hostInput2[i];		
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));
  
  struct timeval timekernel;
  timeStart(&timekernel);
  for (int i=0; i<nStreams; ++i){
  	int offset = i*streamSize;
    //@@ Insert code to below to Copy memory to the GPU here
  	cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], 
  				streamBytes, cudaMemcpyHostToDevice, stream[i]);
  	cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], 
  				streamBytes, cudaMemcpyHostToDevice, stream[i]);
    
    //@@ Initialize the 1D grid and block dimensions here
    int TPB = atoi(argv[2]);
    int BPG = (streamSize + TPB - 1)/TPB; 

    //@@ Launch the GPU Kernel here
    // mykernel<<<blocks, threads, shared_mem, stream>>>(args);
    vecAdd<<<BPG, TPB, 0, stream[i]>>>
    (&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],
  					streamBytes, cudaMemcpyDeviceToHost, stream[i]);
  }
  cudaDeviceSynchronize();
  timeStop(&timekernel);

  //@@ Insert code below to compare the output with the reference
  int equal = 1;
  for(int i=0; i<inputLength; i++){
  	if (hostOutput[i] == resultRef[i]){
  		continue;
  	}
    else{
      printf("The result is not equal in %dth vector element.\n",i);
      printf("hostOutput: %lf and reference: %lf", hostOutput[i], resultRef[i]);
      equal = 0;
    }
  }
  if (equal){
    printf("The output is equal to the reference");
  } 
  	

  //@@ Free the GPU memory here
  for (int i=0; i<nStreams; i++){
  	cudaStreamDestroy(stream[i]);
  }
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  
  return 0;
}
