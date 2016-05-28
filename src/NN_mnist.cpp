#include "NN_mnist.h"

#include <iostream>
#include <fstream>
#include <assert.h>

namespace NN
{

MNIST::MNIST() :
  data(0),
  label(0),
  img_count(0),
  img_row(0),
  img_col(0)
{
}

MNIST::~MNIST()
{
  delete[] data;
  delete[] label;
}

void MNIST::loadTrainData()
  {
    // image
    {
      std::ifstream ifs("../data/train-images.idx3-ubyte", std::ios::in | std::ios::binary);
      assert(!ifs.fail());
      int ival = 0;
      ifs.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      assert(2051); // magic number

      ifs.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      img_count = ival;

      ifs.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      img_row = ival;

      ifs.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      img_col = ival;

      data = new std::vector<float>[img_count];
      // each image
      const size_t sz = img_row * img_col;
      unsigned char* bt = new unsigned char[sz];
      float* fl = new float[sz];
      for (int di = 0; di < img_count; ++di) {
        ifs.read((char*)bt, sizeof(unsigned char) * sz);
        for (int i = 0; i < sz; ++i) {
          fl[i] = bt[i] / 255.0f;
        }
        data[di].assign(fl, fl + sz);
      }
      delete[] fl;
      delete[] bt;
    }

    // label
    {
      std::ifstream ifs("../mnist/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);

      assert(!ifs.fail());
      int ival = 0;
      ifs.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      assert(2051); // magic number

      ifs.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      img_count = ival;
//      assert(img_count == ival);

      unsigned char* bt = new unsigned char[img_count];
      ifs.read((char*)bt, sizeof(bt[0])*img_count);

      label = new std::vector<float>[img_count];
      float tbl[10][10] = 
      {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      };

      for (int i = 0; i < img_count; ++i) {
        int id = bt[i];
        label[i].assign(tbl[id], tbl[id] + 10);
      }
    }
  }

int MNIST::reverseByte(int num)
{
  char bt[4];
  char* pt = (char*)(&num);

  bt[3] = *pt++;
  bt[2] = *pt++;
  bt[1] = *pt++;
  bt[0] = *pt;

  return *(int*)(bt);
}
}