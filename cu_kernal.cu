
// 1、利用好memory是cuda中高性能的关键点
// 2、主要理解pinned memory、global memory、shared memory即可，其他不常用
// 3、GPU可以直接访问pinned memory而不能访问pageable memory
// 4、new、malloc分配的，是pageable memory,由cudaMallocHost分配的是PinnedMemory,由cudaMalloc分配的是GlobalMemory
// 5、尽量多用PinnedMemory储存host数据，或者显式处理Host到Device时，用PinnedMemory做缓存，都是提高性能的关键

// 1、共享内存因为更靠近计算单元，所以访问速度更快
// 2、共享内存通常可以作为访问全局内存的缓存使用
// 3、可以利用共享内存实现线程间的通信
// 4、通常与__syncthreads同时出现，这个函数是同步block内的所有线程，全部执行到这一行才往下走
// 5、使用方式，通常是在线程id为0的时候从global memory取值，然后syncthreads,然后再使用

// 1、核函数是cuda编程的基础
// 2、通过xxx.cu创建一个cudac程序文件，并把cu交给nvcc编译，才能识别cuda语法
// 3、__global__表示为核函数，由host调用。__device__表示为设备函数，由device调用
// 有些printf等常用函数，在nvidia里面封装好了，不用__device__
// __host__表示为主机函数，由host调用。__shared__表示变量为共享变量
// 想要device和host都可以调用，那就在函数前面加上__global__   __host__

// 5、host调用核函数：function<<<gridDim,blockDim,sharedMemorySize,stream>>>(args...);
// gridDim,blockDim告诉cuda要启动多少个线程
dim3 gridDim;
dim3 blockDim;
int nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
// gridDim(21亿，65536，65536)   //runtime API   deviceQuery查询上限
// blockDim（1024，64，64）    blockDim.x * blockDim.y * blockDim.z <=1024

/*    dims                 indexs
    gridDim.z            blockIdx.z
    gridDim.y            blockIdx.y
    gridDim.x            blockIdx.x
    blockDim.z           threadIdx.z
    blockDim.y           threadIdx.y
    blockDim.x           threadIdx.x

    Pseudo code:
    position = 0
    for i in 6:
        position *= dims[i]
        position += indexs[i]
*/

// 只有__global__修饰的函数才可以用<<<>>>的方式调用
// 7、调用核函数是传值的，不能传引用，可以传递类、结构体等，核函数可以是模板(是C++超集，返回值一定返回值void)
// 8、核函数的执行，是异步的，也就是立即返回的
// 9、线程layout主要用到blockDim、gridDim

// 10、核函数内访问线程索引主要用到threadidx、blockidx、blockDim、gridDim这些内置变量