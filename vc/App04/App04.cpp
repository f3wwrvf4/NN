// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include <windows.h>


#include "NN_net.h"
#include "NN_mnist.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <new>
#include <string>
#include <list>

#include <stdlib.h> 
#include <iostream> 
#include <new> 

#if 0

static size_t newCount = 0; 
void* _cdecl operator new(size_t size) 
{
   ++newCount; 
   void* addr = malloc(size); 
   std::cout << "(" << newCount << ") " << "new " << size << "bytes addr 0x" << std::hex << addr << std::dec << std::endl; 
   return addr;
} 
void  _cdecl operator delete(void *p) 
{ 
   --newCount; 
   std::cout << "(" << newCount << ") " << "delete " << p << std::endl; 
   free(p); 
} 

#endif


int main()
{
  int layer_num = 5;
  int node_num[] = { 3, 5, 3, 4,  2 };

  NN::Network net(layer_num, node_num, 8);

  net.train();
  net.save("nand.nn");

  NN::Vector in(3);
  in[0] = 1, in[1] = 1, in[2] = 0;
  std::cout << net.eval(in);

  getchar();
  return 0;
}
