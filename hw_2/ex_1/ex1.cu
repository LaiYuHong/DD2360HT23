#include <stdio.h>
#include <sys/time.h>

#define DataType double


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

//@@ Insert code to implement timer stop


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

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *) malloc(inputLength*sizeof(DataType));
  hostInput2 = (DataType *) malloc(inputLength*sizeof(DataType));
  hostOutput = (DataType *) malloc(inputLength*sizeof(DataType));
  resultRef = (DataType *) malloc(inputLength*sizeof(DataType));
  
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

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, hostOutput, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  int TPB = atoi(argv[2]);
  int BPG = (inputLength + TPB - 1)/TPB;  

  //@@ Launch the GPU Kernel here
  vecAdd<<<BPG, TPB>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

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
