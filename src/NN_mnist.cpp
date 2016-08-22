#include "NN_mnist.h"

#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>

namespace NN
{

void MNIST::LoadTrainData(Content& data)
{ 
  loadData(
    "../data/train-images.idx3-ubyte",
    "../data/train-labels.idx1-ubyte",
    data);
}

void MNIST::LoadTestData(Content& data)
{ 
  loadData(
    "../data/t10k-images.idx3-ubyte", 
    "../data/t10k-labels.idx1-ubyte", 
    data);
}

void MNIST::loadData(const char* data_path, const char* label_path, Content& dst)
{
  int img_count;
  int img_row;
  int img_col;

  // image
  {
    std::ifstream ifs_img(data_path, std::ios::in | std::ios::binary);
    assert(!ifs_img.fail());
    int ival = 0;
    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);

    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);
    img_count = ival;

    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);
    img_row = ival;

    ifs_img.read((char*)(&ival), sizeof(ival));
    ival = reverseByte(ival);
    img_col = ival;

    dst.reserve(img_count);

    // label
    unsigned char* labels = new unsigned char[img_count];
    {
      std::ifstream ifs_lbl(label_path, std::ios::in | std::ios::binary);

      assert(!ifs_lbl.fail());
      int ival = 0;
      ifs_lbl.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);

      ifs_lbl.read((char*)(&ival), sizeof(ival));
      ival = reverseByte(ival);
      assert(img_count == ival);
      ifs_lbl.read((char*)labels, sizeof(labels[0])*img_count);
      ifs_lbl.close();
    }

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

    // each image
    const size_t sz = img_row * img_col;
    unsigned char* bt = new unsigned char[sz];
    float* fl = new float[sz];
    for (int di = 0; di < img_count; ++di) {
      ifs_img.read((char*)bt, sizeof(unsigned char) * sz);
      for (int i = 0; i < sz; ++i) {
        fl[i] = bt[i] / 255.0f;
      }
      int d = (int)labels[di];
      dst.push_back( fl, tbl[d] );
    }
    delete[] fl;
    delete[] bt;
  }
}

}